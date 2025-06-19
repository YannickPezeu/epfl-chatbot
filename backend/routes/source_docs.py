from fastapi import HTTPException, APIRouter
from myUtils.connect_acad2 import initialize_all_connection
from starlette.responses import StreamingResponse
import io

router = APIRouter(
    prefix="/source_docs",
    tags=["source_docs"]  # This will group your library endpoints in the FastAPI docs
)


def get_source_doc_blob_from_db_online(source_doc_id):
    conn, cursor = initialize_all_connection()
    cursor.execute("SELECT file FROM source_docs WHERE id=%s", (source_doc_id,))
    source_doc_blob = cursor.fetchone()[0]
    conn.close()
    return source_doc_blob


@router.get("/view-source-doc/{source_doc_id}")
async def view_source_doc(source_doc_id: int, doc_type: str='pdf'):
    # Fetch the source_doc blob from the database using the provided ID
    source_doc_id = int(source_doc_id)
    source_doc_blob = get_source_doc_blob_from_db_online(source_doc_id)
    if not source_doc_blob:
        raise HTTPException(status_code=404, detail="source_doc not found")

    if doc_type == 'pdf':
        return StreamingResponse(io.BytesIO(source_doc_blob), media_type='application/pdf',
                             headers={"Content-Disposition": f"inline; filename=database_file_{source_doc_id}.pdf"})

    else:
        print('doc_type not recognized')
        raise Exception


