{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "3d0dab33-46f8-41fd-8912-d497f85c4859",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\cdant\\anaconda3\\envs\\jupyter_clean\\Lib\\site-packages\\tqdm\\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html\n",
      "  from .autonotebook import tqdm as notebook_tqdm\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "* Running on local URL:  http://127.0.0.1:7861\n",
      "* To create a public link, set `share=True` in `launch()`.\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div><iframe src=\"http://127.0.0.1:7861/\" width=\"100%\" height=\"500\" allow=\"autoplay; camera; microphone; clipboard-read; clipboard-write;\" frameborder=\"0\" allowfullscreen></iframe></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import gradio as gr\n",
    "from sentence_transformers import SentenceTransformer, util\n",
    "import torch\n",
    "import pandas as pd\n",
    "import os\n",
    "\n",
    "# --- Global Variables for Model and Data (will be updated by file upload) ---\n",
    "model = SentenceTransformer(\"all-MiniLM-L6-v2\")\n",
    "faq_data = {}\n",
    "faq_questions = []\n",
    "faq_embeddings = None\n",
    "similarity_threshold = 0.6\n",
    "\n",
    "# --- Function to Load FAQs from CSV or Excel File ---\n",
    "def load_faqs_from_file(file_path):\n",
    "    global faq_data, faq_questions, faq_embeddings\n",
    "\n",
    "    if file_path is None:\n",
    "        return \"Please upload a file first.\"\n",
    "\n",
    "    # Use pandas to read both csv and excel formats\n",
    "    try:\n",
    "        if file_path.endswith('.csv'):\n",
    "            df = pd.read_csv(file_path)\n",
    "        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):\n",
    "            df = pd.read_excel(file_path)\n",
    "        else:\n",
    "            return \"Unsupported file type. Please upload a CSV or Excel file.\"\n",
    "        \n",
    "        # Ensure the required columns exist (adjust names if necessary)\n",
    "        if 'Question' not in df.columns or 'Answer' not in df.columns:\n",
    "            return \"File must contain 'Question' and 'Answer' columns.\"\n",
    "\n",
    "        # Process data into the required format\n",
    "        new_faq_data = dict(zip(df['Question'], df['Answer']))\n",
    "        new_faq_questions = list(new_faq_data.keys())\n",
    "        \n",
    "        # Check if we actually loaded data\n",
    "        if not new_faq_questions:\n",
    "            return \"Loaded file but found no valid questions and answers.\"\n",
    "\n",
    "        # Generate embeddings for the new data\n",
    "        new_faq_embeddings = model.encode(new_faq_questions, convert_to_tensor=True)\n",
    "\n",
    "        # Update global variables\n",
    "        faq_data = new_faq_data\n",
    "        faq_questions = new_faq_questions\n",
    "        faq_embeddings = new_faq_embeddings\n",
    "        \n",
    "        # Clean up the temporary uploaded file\n",
    "        if os.path.exists(file_path):\n",
    "            os.remove(file_path)\n",
    "\n",
    "        return f\"Successfully loaded {len(faq_questions)} FAQs from the file.\"\n",
    "\n",
    "    except Exception as e:\n",
    "        return f\"Error loading file: {e}\"\n",
    "\n",
    "# --- Chatbot Logic Function ---\n",
    "def ask_chatbot(user_query):\n",
    "    # Check if data is loaded\n",
    "    if not faq_embeddings is not None and len(faq_questions) > 0:\n",
    "         return \"FAQs are not yet loaded. Please upload a file first using the button above.\"\n",
    "\n",
    "    if not user_query.strip():\n",
    "        return \"Please enter a question.\"\n",
    "    \n",
    "    query_embedding = model.encode(user_query, convert_to_tensor=True)\n",
    "    cosine_scores = util.cos_sim(query_embedding, faq_embeddings)\n",
    "    \n",
    "    best_match_index = torch.argmax(cosine_scores).item()\n",
    "    best_score = cosine_scores[0, best_match_index].item()\n",
    "    \n",
    "    if best_score >= similarity_threshold:\n",
    "        best_question = faq_questions[best_match_index]\n",
    "        return faq_data[best_question]\n",
    "    else:\n",
    "        return \"Sorry, I couldn't find a relevant answer in my FAQs.\"\n",
    "\n",
    "# --- Gradio Interface Setup using Blocks for more flexibility ---\n",
    "with gr.Blocks(title=\"FAQ Chatbot with Upload\") as demo:\n",
    "    gr.Markdown(\"# FAQ Chatbot\")\n",
    "    gr.Markdown(\"Upload your 'Question' and 'Answer' CSV/Excel file below to update the knowledge base.\")\n",
    "\n",
    "    # Upload Component\n",
    "    upload_button = gr.UploadButton(\n",
    "        \"Click to Upload CSV/XLSX File\", \n",
    "        file_types=[\".csv\", \".xls\", \".xlsx\"]\n",
    "    )\n",
    "    # Output box for upload status messages\n",
    "    upload_status = gr.Textbox(label=\"Upload Status\", value=\"Ready. No FAQs loaded yet.\")\n",
    "\n",
    "    # Link the upload button action to the loading function\n",
    "    # The uploaded file path is passed as the input to load_faqs_from_file\n",
    "    # The output updates the upload_status textbox\n",
    "    upload_button.upload(\n",
    "        fn=load_faqs_from_file, \n",
    "        inputs=upload_button, \n",
    "        outputs=upload_status\n",
    "    )\n",
    "    \n",
    "    # Chatbot components\n",
    "    chatbot_input = gr.Textbox(placeholder=\"Ask a question here...\")\n",
    "    chatbot_output = gr.Textbox(label=\"Chatbot Answer\")\n",
    "    \n",
    "    # Button to trigger the chatbot function manually if needed\n",
    "    gr.Button(\"Ask\").click(\n",
    "        fn=ask_chatbot,\n",
    "        inputs=chatbot_input,\n",
    "        outputs=chatbot_output\n",
    "    )\n",
    "\n",
    "# --- Launch the App ---\n",
    "if __name__ == \"__main__\":\n",
    "    # Remove the hardcoded initial data load\n",
    "    demo.launch()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f0ae7ae9-9ab5-416e-9932-9f029c3fb9e5",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.14"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
