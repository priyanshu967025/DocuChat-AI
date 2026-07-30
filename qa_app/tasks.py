import logging
from celery import shared_task
from .models import Document, AnalyticsMetric
from .utils import process_document, extract_text_with_ocr

logger = logging.getLogger(__name__)


@shared_task
def process_document_async(doc_id):
    """
    Celery background task for processing uploaded PDF documents (chunking, vectorization).
    """
    try:
        doc = Document.objects.get(pk=doc_id)
        doc.pdf_file.open('rb')
        success = process_document(doc, doc.pdf_file)
        doc.pdf_file.close()

        if success:
            AnalyticsMetric.objects.create(
                user=doc.user,
                event_type='vector_index',
                status='success'
            )
            logger.info(f"Celery task successfully processed document {doc_id}")
            return True
    except Exception as e:
        logger.error(f"Celery document processing failed for {doc_id}: {e}")
        return False


@shared_task
def run_ocr_async(doc_id):
    """
    Celery background task for OCR text extraction on scanned PDFs.
    """
    try:
        doc = Document.objects.get(pk=doc_id)
        ocr_text = extract_text_with_ocr(doc)
        if ocr_text:
            doc.is_ocr_processed = True
            doc.full_text = ocr_text
            doc.save()
            AnalyticsMetric.objects.create(
                user=doc.user,
                event_type='ocr',
                status='success'
            )
            logger.info(f"Celery task completed OCR for document {doc_id}")
            return True
    except Exception as e:
        logger.error(f"Celery OCR processing failed for {doc_id}: {e}")
        return False
