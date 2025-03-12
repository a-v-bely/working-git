import base64
import torch
from torch import nn
import gradio as gr
from collections import defaultdict
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np
import ollama
from pymorphy3 import MorphAnalyzer


# CSS for LaTeX. Still problems
custom_css = """
mjx-container[jax="CHTML"][display="true"] {
    background: #f8f9fa !important;
    padding: 20px !important;
    border-radius: 8px !important;
    margin: 15px 0 !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    overflow-x: auto !important;
}

mjx-container[jax="CHTML"] {
    margin: 10px 0 !important;
}
"""

mathjax_script = """
<script>
MathJax = {
    tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']]
    }
};
</script>
"""

# Initialize Sentence Transformer for embeddings
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
ner_model_name = "denis-gordeev/rured2-ner-microsoft-mdeberta-v3-base"
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
morph = MorphAnalyzer()

# Initialize OpenAI-like API client for Ollama
SYSTEM_PROMPTS = {
    'context': 'Ты - лучший автоматический ассистент. Твоя задача ответить на вопрос пользователя используя только информацию из предоставленных документов. Отвечай подробно, но только на основе документов. Если в документах не содержится полезная информация, необходимая для ответа на вопрос, так и скажи. Не пытайся вспомнить или придумать ответ самостоятельно.',
    'no_context': 'Ты - лучший автоматический ассистент. Ты отвечаешь всегда точно, отмечаешь все ключевые моменты, но при этом достаточно лаконично (если не указано другого).',
    'math': 'Ты - лучший автоматический ассистент по математике. Ты решаешь задачи и даешь подробные комментарии. В начале и в конце каждого математического выражения (формулы) пиши символы "$$".',
    'vision': 'Ты - лучший автоматический ассистент.'
    }

def basic_ner_predict(text:str):
    sigmoid = nn.Sigmoid()
    tokenized = ner_tokenizer(text)
    input_ids = torch.tensor([tokenized["input_ids"]], dtype=torch.long)
    token_type_ids = torch.tensor([tokenized["token_type_ids"]], dtype=torch.long)
    attention_mask = torch.tensor([tokenized["attention_mask"]], dtype=torch.long)
    preds = ner_model(**{"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask})
    logits = sigmoid(preds.logits)
    output_tokens = []
    output_preds = []
    id_to_label = {int(k): v for k, v in ner_model.config.id2label.items()}
    for i, token in enumerate(input_ids[0]):
        if token > 3:
            class_ids = (logits[0][i] > 0.5).nonzero()
            if class_ids.shape[0] >= 1:
                class_names = [id_to_label[int(cl)] for cl in class_ids]
            else:
                class_names = [id_to_label[int(logits[0][i].argmax())]]
            converted_token = ner_tokenizer.convert_ids_to_tokens([token])[0]
            new_word_bool = converted_token.startswith("▁")
            converted_token = converted_token.replace("▁", "")
            if not(new_word_bool) and output_tokens:
                output_tokens[-1] += converted_token
            else:
                output_tokens.append(converted_token)
                output_preds.append(class_names)
        else:
            class_names = []
    output_togetger = [[output_tokens[t_i], output_preds[t_i]] for t_i in range(len(output_tokens))]   
    return output_togetger

def clear_annotations(tags):
    seen_ner_types = set()
    result = []
    for tag in sorted(tags):
        # Extract the B/I indicator and the NER type
        if tag == 'O':
            result.append(tag)
            continue
        indicator, ner_type = tag.split('-')
        if ner_type in seen_ner_types:
            # If the NER type has already been seen, only keep 'I'
            if indicator == 'I':
                result = [t for t in result if not t.endswith(f"-{ner_type}")]
                result.append(tag)
        else:
            # If the NER type hasn't been seen yet, add it to the result
            seen_ner_types.add(ner_type)
            result.append(tag)
    return result

def extract_ner_sequences(annotated_tokens):
    sequences = []
    # Dictionary to map NER tags to their corresponding sequences
    ner_sequences = {}
    for token_info in annotated_tokens:
        token = token_info[0]
        token = ''.join(l for l in token if l.isalnum())
        token = morph.parse(token)[0].normal_form
        tags = token_info[1]
        # Skip tokens with 'O' as the only tag
        if all(tag == 'O' for tag in tags):
            continue
        # Clear annotations if two tags with different B/I indicators occur together
        tags = clear_annotations(tags)
        # Process each tag for the current token
        for tag in tags:
            if tag == 'O':
                continue
            # Extract the NER type from the tag (e.g., 'B-COUNTRY' -> 'COUNTRY')
            ner_type = tag.split('-')[1].replace('GPE', 'GEO-POLITCS').replace('FAC', 'FACT').replace('NORP', 'GROUP')
            # Check if the sequence for this NER type is already started
            if ner_type not in ner_sequences:
                ner_sequences[ner_type] = []
            # Determine if the token is a beginning ('B') or inside ('I')
            if tag.startswith('B'):
                # Start a new sequence
                ner_sequences[ner_type].append([token])
            elif tag.startswith('I'):
                # Add to the last sequence of this NER type if it exists
                if ner_sequences[ner_type]:
                    ner_sequences[ner_type][-1].append(token)
                else:
                    # If there's no ongoing sequence, start a new one
                    ner_sequences[ner_type].append([token])
    # Combine the sequences into the desired format
    for ner_type, sequences_list in ner_sequences.items():
        for sequence in sequences_list:
            sequences.append([" ".join(sequence), ner_type])
    return sequences

def top_level_ner(documents):
    # print(documents)
    if not documents:
        return 'Документы не загружены.'
    NER_SEQS_DICT = defaultdict(dict)
    for doc in documents:
        doc = doc.page_content
        basic_ner = basic_ner_predict(doc)     
        ner_seqs = extract_ner_sequences(basic_ner)
        for seq, ner in ner_seqs:
            if seq in NER_SEQS_DICT[ner]:         
                NER_SEQS_DICT[ner][seq] += 1
            else:
                NER_SEQS_DICT[ner][seq] = 1

    NER_SEQS_DICT = [(k, sorted(((kk, vv) for kk, vv in v.items()), key=lambda x: x[1], reverse=True)) for k, v in NER_SEQS_DICT.items()]
    out = ''
    for x, y in NER_SEQS_DICT:
        out += f'# {x}\n\t- '
        out += '\n\t- '.join([k for k, v in y]).strip() + '\n\n'
    return out

def llm_extract_ner(text, selected_model):
    messages = [
        {'role': 'system', 'content': 
         """Ты автоматический ассистент. Твоя задача извлечь именованные сущности из текста. 
         Сущности, которые необходимо найти, относятся к категориям которые перечислены ниже.
         Для каждой категории извлеки все соответствующие сущности, упомянутые в тексте, и верни результат в формате JSON, 
         где ключами будут типы сущностей, a а значениями — множества уникальных названий сущностей.  
         Категории именованных сущностей: 
         age, application, award, borough, cardinal, character, city, commodity, country, crime, currency, date, disease, economic-sector, 
         event, fac, family, geo-political-entity, group, house, investment-program, language, law, location, money, nationality, 
         news-source, norp (nationalities or religious or political groups), ordinal, organization, penalty, percent, person, price, product, 
         profession, quantity, region, religion, street, time, trade-agreement, vehicle, village, weapon, website, work-of-art
         Пример входного текста:
         'Иван путешествовал из Москвы в Санкт-Петербург прошлым летом. Он планирует посетить Париж в следующем году.'
         Cтруктура ожидаемого вывода:
         {
         "PERSON": {"Иван", "Он"},
         "CITY": {"Москва", "Санкт-Петербург", "Париж"},
         "TIME": {"прошлое лето", "в следующий год"}
         }
         """}, 
        {'role': 'user', 'content': f'Извлеки именованные сущности из текста: {text}'}]
    response = ollama.chat(
        model=selected_model,
        messages=messages,
        options={'temperature': 0.3, 'max_tokens': 4096},
        format='json'
    )
    return response.message.content

def top_level_llm_ner(documents, llm_select):
    if not documents:
        return 'Документы не загружены.'
    llm_select = {"LLAMA":"llama3.3:latest", 
            "DEEPSEEK": "deepseek-r1:70b", 
            "LLAMA IMG": "llama3.2-vision:11b-instruct-fp16", 
            "MATH": "rscr/ruadapt_qwen2.5_32b:Q8_0"}.get(llm_select)
    NER = []
    for doc in documents[:2]:
        doc = doc.page_content
        ner = llm_extract_ner(doc, llm_select)
        print(ner)
        print('='*50)
        NER.append(ner)
    return NER
    
# Function to load documents from directory or single file
def load_documents(file_paths):
    documents = []
    for file_path in file_paths:
        if file_path.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.doc'):
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith(".csv"):
            loader = CSVLoader(file_path)
        elif file_path.endswith(('.htm', '.html')):
            loader = UnstructuredHTMLLoader(file_path)
        elif file_path.endswith('.md'):
            loader = UnstructuredMarkdownLoader(file_path)
        elif file_path.endswith('.odt'):
            loader = UnstructuredODTLoader(file_path)
        elif file_path.endswith(('.ppt', '.pptx')):
            loader = UnstructuredPowerPointLoader(file_path)
        else:
            continue  # Skip unsupported file types
        documents.extend(loader.load())
    return documents

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

# Function to load images from directory or single file
def load_images(file_paths):
    images = []
    for file_path in file_paths:
        if file_path.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.ico', '.svg')):
            endoded_image = encode_image(file_path)
            images.append((file_path, endoded_image))
    return images

# Function to split documents into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    return text_splitter.split_documents(documents)

# Function to generate embeddings for documents
def generate_embeddings(documents):
    texts = [doc.page_content for doc in documents]
    embeddings = embedding_model.encode(texts)
    return embeddings

# Function to retrieve relevant documents based on query
def retrieve_relevant_documents(query, documents, embeddings, top_k=100):
    query_embedding = embedding_model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    relevant_docs = [documents[i] for i in top_indices]
    return relevant_docs

# Main function to handle user input and generate responses
def chat_with_rag(query, selected_model, chat_history, documents, embeddings, images, log_text):
    selected_model = {"LLAMA":"llama3.3:latest", 
                    "DEEPSEEK": "deepseek-r1:70b", 
                    "LLAMA IMG": "llama3.2-vision:11b-instruct-fp16", 
                    "MATH": "rscr/ruadapt_qwen2.5_32b:Q8_0"}.get(selected_model)
    # Check if any documents are loaded
    if documents:
        # Retrieve relevant documents
        relevant_docs = retrieve_relevant_documents(query, documents, embeddings)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Log relevant documents in the logging window
        if relevant_docs:
            log_message = f"Нашел {len(relevant_docs)} релевантных контекстов."
            # log_message += "\n\n".join([f"- {doc.page_content[:200]}..." for doc in relevant_docs])  # Show first 200 chars of each doc
        else:
            log_message = "Релевантных контекстов не найдено."
    else:
        # No documents loaded, proceed without context
        context = ""
        log_message = "Документы не загружены."
    
    # Construct user history
    if chat_history:
        user_history = chat_history
    else:
        user_history = []
        
    if context:
        SYS_MSG = {'role': 'system', 'content': SYSTEM_PROMPTS['context']}
        if user_history and not any(msg['role']=='system' for msg in user_history):
            user_history.insert(0, SYS_MSG)
        elif user_history and any(msg['role']=='system' for msg in user_history):
            for i, msg in enumerate(user_history):
                if msg['role'] == 'system':
                    user_history[i] = SYS_MSG
        else:
            user_history = [SYS_MSG, ]
        user_history.append({'role': 'user', 'content': f'Документы:\n{context}\n\nВопрос пользователя:\n{query}.'})
    elif images and 'vision' in selected_model:
        SYS_MSG = {'role': 'system', 'content': SYSTEM_PROMPTS['vision']}
        if user_history and not any(msg['role']=='system' for msg in user_history):
            user_history.insert(0, SYS_MSG)
        elif user_history and any(msg['role']=='system' for msg in user_history):
            for i, msg in enumerate(user_history):
                if msg['role'] == 'system':
                    user_history[i] = SYS_MSG
        else:
            user_history = [SYS_MSG, ]

        img_descriptions = []
        for img_path, image in images:
            response = ollama.chat(
                model='llama3.2-vision:11b-instruct-q4_K_M',
                messages=[{'role': 'user',
                        'content': f'Подробно опиши изображение. Обрати особое внимание на {query}',
                        'images': [image]}],
                options={'temperature': 0.3,}
                )
            img_descriptions.append((img_path, response.message.content))
            
        user_history.append({'role': 'user', 'content': f'Ответь на вопрос и обязательно укажи соответствующие пути к файлам.\n\nВопрос:\n{query}.\n\nИнформация об изображениях:\n{img_descriptions}'})
    elif not context and 'qwen' in selected_model:
        SYS_MSG = {'role': 'system', 'content': SYSTEM_PROMPTS['math']}
        if user_history and not any(msg['role']=='system' for msg in user_history):
            user_history.insert(0, SYS_MSG)
        elif user_history and any(msg['role']=='system' for msg in user_history):
            for i, msg in enumerate(user_history):
                if msg['role'] == 'system':
                    user_history[i] = SYS_MSG
        else:
            user_history = [SYS_MSG, ]
        user_history.append({'role': 'user', 'content': query})
    else:
        SYS_MSG = {'role': 'system', 'content': SYSTEM_PROMPTS['no_context']}
        if user_history and not any(msg['role']=='system' for msg in user_history):
            user_history.insert(0, SYS_MSG)
        elif user_history and any(msg['role']=='system' for msg in user_history):
            for i, msg in enumerate(user_history):
                if msg['role'] == 'system':
                    user_history[i] = SYS_MSG
        else:
            user_history = [SYS_MSG, ]
        user_history.append({'role': 'user', 'content': query})

    # print('='*50+'\n'+'BEFORE USER HISTORY'+'\n'+'='*50+'\n', user_history)
    # print('='*50+'\n'+'BEFORE CHAT HISTORY'+'\n'+'='*50+'\n', chat_history, '\n')

    # Generate response using OpenAI-like API
    response = ollama.chat(
        model=selected_model,
        messages=user_history,
        options={'temperature': 0.3, 'max_tokens': 4096}
    )
    if images and 'vision' in selected_model:
        user_history[-1] = {'role': 'user', 'content': query}
    if context:
        user_history[-1] = {'role': 'user', 'content': query}

    # Extract the answer from the response
    answer_str = response.message.content\
        .replace('($', '$$').replace('$)', '$$')\
        .replace('\\[', '$$').replace('\\]', '$$')\
        .replace('\\(', '$$').replace('\\)', '$$')\
        .replace('<think>', '\n**Размышляю**:\n')\
        .replace('</think>', '\n**Закончил думать. Ответ:**\n')
   
    # Update chat history in the correct format
    if not chat_history or (chat_history and chat_history[-1]['role'] != 'user'):
        chat_history.append({'role': 'user', 'content': query})
    chat_history.append({'role': 'assistant', 'content': answer_str})
    
    # Return updated chat history and log message
    # print('='*50+'\n'+'AFTER USER HISTORY'+'\n'+'='*50+'\n', user_history)
    # print('='*50+'\n'+'AFTER CHAT HISTORY'+'\n'+'='*50+'\n', chat_history, '\n')
    return chat_history, log_message

# Function to clear chat history
def clear_chat():
    return [], []  # Return empty lists for both chatbot display and chat history state

# Function to load documents when "Load Documents" button is clicked
def load_uploaded_documents(uploaded_files):
    # print(uploaded_files)
    # Load and process documents
    documents = load_documents(uploaded_files)
    split_docs = split_documents(documents)
    embeddings = generate_embeddings(split_docs)
    images = load_images(uploaded_files)
    
    # Log the number of chunks created
    log_message = f"Документы и изображения успешно загружены. Создана база данных из {len(split_docs)} текстовых контекстов и {len(images)} изображений."
    return split_docs, images, embeddings, log_message

# Function to clear document storage
def clear_document_storage():
    # Clear documents and embeddings
    log_message = "Файлы и изображения удалены."
    return [], [], [], [], [], log_message

# Function to export chat history to a .txt file
def export_chat_history(chat_history):
    chat_text = ""
    for message in chat_history:
        if message['role'] == 'system':
            continue
        role = message['role'].capitalize()  # "User" or "Assistant"
        content = message['content']
        chat_text += f"{role}: {content}\n\n"
    return chat_text

# Gradio Interface
with gr.Blocks(css=custom_css, head=mathjax_script) as demo:
    gr.HTML("<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.min.js'></script>")
    gr.Markdown("# Ассистент")
    
    # State to store uploaded files, documents, embeddings, chat history, and log text
    uploaded_files_paths_state = gr.State([])
    uploaded_files_state = gr.State([])
    documents_state = gr.State([])
    images_state = gr.State([])
    embeddings_state = gr.State([])
    chat_history_state = gr.State([])
    log_text_state = gr.State("")
    
    with gr.Row():
        with gr.Column(scale=1):
            # LLM selection dropdown
            llm_select = gr.Dropdown(choices=["LLAMA", "DEEPSEEK", "LLAMA IMG", "MATH"], label="Выберите языковую модель", value="LLAMA")
            file_upload = gr.File(label="Загрузить документ(-ы)", 
                                    file_types=[".txt", ".pdf", ".doc", ".docx", ".csv", '.htm', '.html', '.md', '.odt', '.ppt',
                                                '.pptx', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', 
                                                '.ico', '.svg'], 
                                    file_count="multiple", scale=1)
            with gr.Row(scale=1):
                load_button = gr.Button("Загрузить")
                clear_documents_button = gr.Button("Очистить")
            # Logging window
            log_text = gr.Textbox(label="Логи", lines=2, interactive=False, scale=2)
        with gr.Column(scale=3):
            # Chat window
            chatbot = gr.Chatbot(label="Диалог", type="messages", render_markdown=True, height=450, show_copy_button=True, 
                                )  # Set type='messages'
            query_input = gr.Textbox(label="Отправить сообщение")
            # Send question & clear history buttons
            with gr.Row():
                submit_button = gr.Button("Отправить")
                clear_button = gr.Button("Очистить")
                export_button = gr.DownloadButton("Сохранить диалог")

    with gr.Row():
        with gr.Column():
            ner_window = gr.Textbox(label="Именованные сущности 1", lines=5, interactive=False, autoscroll=False)
            ner_button = gr.Button("Извлечь сущности 1")
        # with gr.Column():
        #     ner_window2 = gr.Textbox(label="Именованные сущности 2", lines=5, interactive=False, autoscroll=False)
        #     ner_button2 = gr.Button("Извлечь сущности 2")

    # Event listeners
    def update_uploaded_files(files, uploaded_files_paths_state):
        res = [file.name for file in files]
        res = list(set(uploaded_files_paths_state+res))
        return res, res
    
    file_upload.upload(update_uploaded_files, inputs=[file_upload, uploaded_files_paths_state], outputs=[uploaded_files_state, uploaded_files_paths_state])
    
    # Load documents button event listener
    load_button.click(load_uploaded_documents, inputs=uploaded_files_state, outputs=[documents_state, images_state, embeddings_state, log_text])

     # Clear document storage button event listener
    clear_documents_button.click(clear_document_storage, inputs=[], outputs=[uploaded_files_paths_state, uploaded_files_state, documents_state, embeddings_state, images_state, log_text])

    def respond(message, model, chat_history, documents, embeddings, images, log_text):
        updated_chat_history, updated_log_text = chat_with_rag(message, model, chat_history, documents, embeddings, images, log_text)
        return "", updated_chat_history, updated_log_text, updated_chat_history, updated_log_text
    
    query_input.submit(respond, inputs=[query_input, llm_select, chat_history_state, documents_state, embeddings_state, images_state, log_text_state], 
                        outputs=[query_input, chat_history_state, log_text, chatbot, log_text_state])
    submit_button.click(respond, inputs=[query_input, llm_select, chat_history_state, documents_state, embeddings_state, images_state, log_text_state], 
                        outputs=[query_input, chat_history_state, log_text, chatbot, log_text_state])
    
    # Clear chat button event listener
    clear_button.click(clear_chat, inputs=[], outputs=[chatbot, chat_history_state])

    # Export chat history button event listener
    def export_chat(chat_history):
        chat_text = export_chat_history(chat_history)
        file_path = "chat_history.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(chat_text)
        return file_path
    
    export_button.click(export_chat, inputs=chat_history_state, outputs=export_button)

    ner_button.click(top_level_ner, inputs=[documents_state], outputs=[ner_window])
    # ner_button2.click(top_level_llm_ner, inputs=[documents_state, llm_select], outputs=[ner_window2])

# Launch the Gradio app
demo.launch(server_name='0.0.0.0')
