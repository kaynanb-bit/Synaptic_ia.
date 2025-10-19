# Synaptic/backend/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sympy import symbols, Eq, solve, sympify
import csv
import io
import os
from openai import OpenAI

# Cria o app FastAPI
app = FastAPI(
    title="SYNAPTIC ENGINEERING AI",
    description="Assistente técnico de engenharia com IA",
    version="1.0.0"
)

# Serve o frontend estático da pasta Synaptic (um nível acima de backend/)
app.mount("/", StaticFiles(directory="../", html=True), name="static")

# Configura CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Remove variáveis de proxy (não são necessárias no Render)
for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    os.environ.pop(key, None)

# Inicializa o cliente OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY não configurada!")

client = OpenAI(api_key=OPENAI_API_KEY)

# Modelos de dados
class ChatRequest(BaseModel):
    messages: list

class EquationRequest(BaseModel):
    equation: str

# Rotas da API
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
        raise HTTPException(status_code=500, detail=f"Erro na OpenAI: {str(e)}")

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
        raise HTTPException(status_code=400, detail=f"Erro na equação: {str(e)}")

@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Apenas arquivos CSV são suportados.")
    try:
        content = await file.read()
        text = content.decode('utf-8')
        reader = csv.DictReader(io.StringIO(text))
        data = list(reader)
        columns = reader.fieldnames or []
        return {
            "filename": file.filename,
            "columns": columns,
            "row_count": len(data),
            "preview": data[:5]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao processar arquivo: {str(e)}")
