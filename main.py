import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from pydantic import BaseModel
from typing import List, Optional

import json
import requests
from pinecone import Pinecone, ServerlessSpec

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
model = AutoModel.from_pretrained("thenlper/gte-large")

class Item(BaseModel):
    model: str
    prompt: str

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def embedding(prompt: str):
    batch_dict = tokenizer([prompt], max_length=512, padding=True, truncation=True, return_tensors='pt')

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    return [float(v) for v in embeddings.detach().numpy()[0]]

@app.post("/")
async def embed(item: Item):
    return embedding(item.prompt)

class Chat(BaseModel):
    model: str
    messages: str

@app.post("/chat")
async def chat(chat: Chat):
    headers = {
        'Content-Type': 'application/json'
    }
    messages = json.loads(chat.messages)
    n = len(messages)
    data = {
            "model": "llama3.1:8b",
            "messages": [
                {
                    "role": "system",
                    "content": "must use user's language, convert all answers according to user's requirement.\
                    - when first inquiry, answer like ```Hi my friend,\nGlad to receive your inquiry.Thank you for your interest in our products, we are Chongqing Haike thermal Insulation Company, we are a professional manufacturer of thermal insulation materials more than 30years.\nWe specialize in the production of Rock Wool sandwich panel, Glass wool, Container house, XPS extruded board, Polyurethane insulation, our store is doing promote activity The more you buy, the more discount for you!\\\nplease feel free to ask me any question```\
                    - when second asking about just catalog, please ask like ```Could you please tell me which product do you want?```\
                    - when talking interesting about product parameters, ask twice about random one parameter in following parameters - `Density`, `Thickness`, `Width`,`Length`, `Tensile strength`, `Compressive strength`, `Levels` like ```Can you tell me the thickness you want.```\
                    - if user answered over just two parameters of product in chat, please answer just only one word `yes`, if not, continue asking about spec. but user start asking new product, please ask again about parameters.\
                    - if user ask questions related to company, also answer just only one word `yescompany`\
                    - just when asking for `freight`, ask like ```As for the freight, could you tell me what shipping terms you would like to use (EXW/FCA/FOB/CIF/DDU/DDP)?```\
                    then when user mention specific type of freight, ask like ```Can you tell me the delivery address you need for shipping so that I can give you the best delivery plan?```\
                    - when ordering the customer needs to ask the customer's name, ask the customer's company name and address, according to this to quote"
                },
            ],
            "stream": False
        }

    for message in messages:
        data["messages"].append(message)

    answer = requests.post("http://localhost:11434/api/chat", headers=headers, data=json.dumps(data)).content
    res = json.loads(answer)["message"]
    
    if res["content"].lower().startswith("yes"):
        pc = Pinecone(api_key="39711f46-9069-4dda-9c28-a4bc6b502229")
        # Create Index
        index_name = "ftozonllama"
        index = pc.Index(index_name)

        if res["content"].lower() == "yescompany":
            response = embedding(messages[-1]["content"])
        else:
            preprompt = {
                "model": "llama3.1:8b",
                "prompt": f"```Using this context: {json.dumps(messages[-6:])}``` let me know only product name and product spec you want, not must include any description",
                "stream": False
            }

            output = json.loads(requests.post("http://localhost:11434/api/generate", data=json.dumps(preprompt)).text) # summarize chat history

            response = embedding(output["response"]) # get vector

        results = index.query( # search from pinecone
            namespace="ns1",
            vector=[response],
            top_k=2,
            include_values=False,
            include_metadata=True
        )
        context = results.to_dict()["matches"][0]["metadata"]["text"] # result

        data = { # generate humanic answer based on context
            "model": "llama3.1:8b",
            "prompt": f"```Using this context: {context}.``` Just Describe like human",
            "stream": False
        }

        output = json.loads(requests.post("http://localhost:11434/api/generate", data=json.dumps(data)).text)
        
        return {"role":"assistant", "content": output["response"]}
    else:
        return res

@app.get("/download/{filename}")
async def download_file(filename: str):
    return FileResponse(f"products/{filename}", filename=filename)

@app.get("/email")
async def send_email():
    import smtplib, ssl

    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = "greendev0317@gmail.com"  # Enter your address
    receiver_email = "chrispalman317@gmail.com"  # Enter receiver address
    password = "hello"
    message = """\
    Subject: Hi there

    This message is sent from Python."""

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)

@app.get("/quote")
async def quote():
    return true