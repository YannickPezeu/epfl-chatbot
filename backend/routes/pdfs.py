from fastapi import HTTPException, APIRouter
from myUtils.connect_acad2 import initialize_all_connection
from starlette.responses import StreamingResponse
import io

router = APIRouter(
    prefix="/pdfs",
    tags=["pdfs"]  # This will group your library endpoints in the FastAPI docs
)


def get_pdf_blob_from_db_online(pdf_id):
    conn, cursor = initialize_all_connection()
    cursor.execute("SELECT file FROM pdfs WHERE id=%s", (pdf_id,))
    pdf_blob = cursor.fetchone()[0]
    conn.close()
    return pdf_blob


@router.get("/view-pdf/{pdf_id}")
async def view_pdf(pdf_id: int):
    # Fetch the PDF blob from the database using the provided ID
    # print('type pdf_id', type(pdf_id))
    pdf_id = int(pdf_id)
    pdf_blob = get_pdf_blob_from_db_online(pdf_id)
    if not pdf_blob:
        raise HTTPException(status_code=404, detail="PDF not found")

    return StreamingResponse(io.BytesIO(pdf_blob), media_type='application/pdf',
                             headers={"Content-Disposition": f"inline; filename=database_file_{pdf_id}.pdf"})

