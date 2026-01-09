{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "172f48bf-c743-4e89-afed-eb7ef690b3ce",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\cdant\\anaconda3\\envs\\jupyter_clean\\Lib\\site-packages\\tqdm\\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html\n",
      "  from .autonotebook import tqdm as notebook_tqdm\n"
     ]
    }
   ],
   "source": [
    "from sentence_transformers import SentenceTransformer, util\n",
    "import torch\n",
    "import json\n",
    "\n",
    "model = SentenceTransformer(\"all-MiniLM-L6-v2\")\n",
    "\n",
    "faq_data = {\n",
    "    \"How does the return policy work?\": \"You can return most items within 30 days of purchase with a receipt.\",\n",
    "    \"What are your store hours this weekend?\": \"We are open from 9 AM to 6 PM on both Saturday and Sunday.\",\n",
    "    \"Can I track my order online?\": \"Yes, use the tracking link provided in your shipping confirmation email.\",\n",
    "    \"How do I reset my password?\": \"Click 'Forgot Password?' on the login page and follow the email instructions.\"\n",
    "}\n",
    "\n",
    "faq_questions = list(faq_data.keys())\n",
    "faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)\n",
    "similarity_threshold = 0.6\n",
    "\n",
    "\n",
    "def handler(request):\n",
    "    try:\n",
    "        body = request.get_json()\n",
    "        user_query = body.get(\"query\", \"\")\n",
    "\n",
    "        query_embedding = model.encode(user_query, convert_to_tensor=True)\n",
    "        cosine_scores = util.cos_sim(query_embedding, faq_embeddings)\n",
    "\n",
    "        best_match_index = torch.argmax(cosine_scores).item()\n",
    "        best_score = cosine_scores[0, best_match_index].item()\n",
    "\n",
    "        if best_score >= similarity_threshold:\n",
    "            answer = faq_data[faq_questions[best_match_index]]\n",
    "        else:\n",
    "            answer = \"Sorry, I couldn't find a relevant answer.\"\n",
    "\n",
    "        return {\n",
    "            \"statusCode\": 200,\n",
    "            \"headers\": {\"Content-Type\": \"application/json\"},\n",
    "            \"body\": json.dumps({\"answer\": answer})\n",
    "        }\n",
    "\n",
    "    except Exception as e:\n",
    "        return {\n",
    "            \"statusCode\": 500,\n",
    "            \"body\": json.dumps({\"error\": str(e)})\n",
    "        }\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c35fc0c7-b954-449b-9e9d-2476dc76b9d5",
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
