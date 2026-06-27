from fastapi import APIRouter, HTTPException
from app.schemas.rag_diag import RagDiagAgentRequest, RagDiagAgentResponse
from app.schemas.ticket import TicketAgentRequest,TicketAgentResponse
from app.services.rag_diag_agent_service import run_rag_diag_agent
from app.services.ticket_agent_service import run_ticket_agent

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/rag-diagnose", response_model=RagDiagAgentResponse)
def rag_diag_agent_api(request: RagDiagAgentRequest):
    try:
        result = run_rag_diag_agent(
            query=request.query,
            doc_id=request.doc_id,
            top_k=request.top_k,
            candidate_k=request.candidate_k,
            rrf_k=request.rrf_k,
            max_chunk_chars=request.max_chunk_chars,
            min_rerank_score=request.min_rerank_score,
            max_context_chars=request.max_context_chars,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to run rag diagnose agent: {e}") from e


@router.post('/ticket',response_model=TicketAgentResponse)
def ticket_agent_api(request:TicketAgentRequest):
    try:
        result=run_ticket_agent(
            query=request.query,
            ticket_id=request.ticket_id
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to run ticket agent: {e}") from e














