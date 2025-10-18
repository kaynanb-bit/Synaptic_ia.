# backend/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sympy import symbols, Eq, solve, sympify
import pandas as pd
import io
import os
from openai import OpenAI

app = FastAPI(
    title="SYNAPTIC ENGINEERING AI API",
    description="Backend para assistente técnico de engenharia com suporte a chat, upload de dados e resolução de equações.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Variável de ambiente OPENAI_API_KEY não configurada!")

client = OpenAI(api_key=OPENAI_API_KEY)

class ChatRequest(BaseModel):
    messages: list

class EquationRequest(BaseModel):
    equation: str

@app.get("/")
def root():
    return {"message": "SYNAPTIC ENGINEERING AI Backend está online!"}

@app.post("/chat")
async def chat_completion(request: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=request.messages,
            max_tokens=3000,
            temperature=0.18
        )
        return {"response": response.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na chamada à OpenAI: {str(e)}")

@app.post("/solve-equation")
async def solve_equation(request: EquationRequest):
    try:
        x = symbols('x')
        eq_str = request.equation.strip()
        if not eq_str:
            raise ValueError("Equação vazia.")
        
        if '=' in eq_str:
            lhs, rhs = eq_str.split('=', 1)
            expr = sympify(lhs.strip()) - sympify(rhs.strip())
        else:
            expr = sympify(eq_str)
        
        solutions = solve(Eq(expr, 0), x)
        return {"solution": str(solutions)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao resolver equação: {str(e)}")

@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Formato de arquivo não suportado. Use CSV ou Excel.")
    
    try:
        content = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))
        
        return {
            "filename": file.filename,
            "columns": df.columns.tolist(),
            "row_count": len(df),
            "preview": df.head(5).to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao processar arquivo: {str(e)}")