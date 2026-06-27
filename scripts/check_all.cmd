@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0.."
set PYTHONPATH=%CD%

echo ==========================================
echo SmartEnv Support System Self Check
echo Project root: %CD%
echo ==========================================

call conda activate pka
if errorlevel 1 (
    echo [FAIL] conda activate pka failed.
    exit /b 1
)

echo.
echo [1/8] Check env...
python scripts\check_env.py
if errorlevel 1 goto fail

echo.
echo [2/8] Check Milvus...
python scripts\check_milvus.py
if errorlevel 1 goto fail

echo.
echo [3/8] Check local models...
python scripts\check_models.py
if errorlevel 1 goto fail

echo.
echo [4/8] Check LLM...
python scripts\check_llm.py
if errorlevel 1 goto fail

echo.
echo [5/8] Check retrieval eval set...
python scripts\check_retrieval_eval_set.py
if errorlevel 1 goto fail

echo.
echo [6/8] Evaluate retrieval...
python scripts\evaluate_retrieval.py --top-k 5
if errorlevel 1 goto fail

echo.
echo [7/8] Evaluate RAG answers...
python scripts\evaluate_rag_answers.py
if errorlevel 1 goto fail

echo.
echo [8/8] Evaluate Support E2E...
echo 注意：这一步需要 FastAPI 后端已启动：http://127.0.0.1:8000
python scripts\evaluate_support_e2e.py
if errorlevel 1 goto fail

echo.
echo ==========================================
echo ALL CHECKS PASSED
echo ==========================================
pause
exit /b 0

:fail
echo.
echo ==========================================
echo CHECK FAILED
echo ==========================================
pause
exit /b 1