import time
from rest_framework import viewsets, permissions, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import action
from django.shortcuts import get_object_or_404
from django.db.models import Sum, Count, Avg
from .models import Document, Conversation, ChatMessage, AnalyticsMetric
from .serializers import (
    DocumentSerializer,
    ConversationSerializer,
    ChatMessageSerializer,
    AnalyticsMetricSerializer
)
from .utils import (
    process_document,
    extract_text_with_ocr,
    retrieve_relevant_chunks,
    generate_answer,
    semantic_faiss_search
)


class DocumentViewSet(viewsets.ModelViewSet):
    """
    DRF ViewSet for Document Upload, Listing, and OCR Extraction.
    """
    serializer_class = DocumentSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Document.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        doc = serializer.save(user=self.request.user)
        # Log upload metric
        AnalyticsMetric.objects.create(
            user=self.request.user,
            event_type='pdf_upload',
            status='success'
        )

    @action(detail=True, methods=['post'])
    def trigger_ocr(self, request, pk=None):
        """
        Runs OCR Text Extraction using PyTesseract on scanned PDFs.
        """
        doc = self.get_object()
        ocr_text = extract_text_with_ocr(doc)
        if ocr_text:
            doc.is_ocr_processed = True
            doc.full_text = ocr_text
            doc.save()
            AnalyticsMetric.objects.create(
                user=request.user,
                event_type='ocr',
                status='success'
            )
            return Response({'status': 'OCR extraction completed', 'text_length': len(ocr_text)})
        return Response({'error': 'OCR extraction failed'}, status=status.HTTP_400_BAD_REQUEST)


class ConversationViewSet(viewsets.ModelViewSet):
    """
    DRF ViewSet for Multi-User Chat Conversations.
    """
    serializer_class = ConversationSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Conversation.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class SemanticSearchAPIView(APIView):
    """
    REST API endpoint for FAISS Vector Search & Chat with Document.
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        doc_id = request.data.get('doc_id')
        conv_id = request.data.get('conv_id')
        question = request.data.get('question', '').strip()

        if not doc_id or not question:
            return Response({'error': 'doc_id and question required'}, status=status.HTTP_400_BAD_REQUEST)

        document = get_object_or_404(Document, pk=doc_id, user=request.user)

        start_time = time.time()

        # FAISS & TF-IDF Semantic Retrieval
        faiss_results = semantic_faiss_search(document, question, top_k=4)
        relevant_chunks = faiss_results if faiss_results else retrieve_relevant_chunks(document, question, top_k=4)

        # Get Answer via Qwen3 / Gemini
        answer = generate_answer(question, relevant_chunks)

        latency_ms = int((time.time() - start_time) * 1000)

        # Save conversation message
        if conv_id:
            conversation = get_object_or_404(Conversation, pk=conv_id, user=request.user)
        else:
            conversation, _ = Conversation.objects.get_or_create(
                user=request.user,
                document=document,
                defaults={'title': f"API Chat - {document.title[:25]}"}
            )

        ChatMessage.objects.create(conversation=conversation, role='user', content=question)
        ai_msg = ChatMessage.objects.create(
            conversation=conversation,
            role='assistant',
            content=answer,
            retrieved_chunks=relevant_chunks[:2],
            latency_ms=latency_ms,
            provider_used='Qwen3/Gemini (FAISS)'
        )

        # Log Analytics Metric
        AnalyticsMetric.objects.create(
            user=request.user,
            event_type='query',
            tokens_used=len(question.split()) + len(answer.split()),
            latency_ms=latency_ms,
            status='success'
        )

        return Response({
            'answer': answer,
            'chunks_used': len(relevant_chunks),
            'latency_ms': latency_ms,
            'conversation_id': conversation.id
        })


class AnalyticsDashboardAPIView(APIView):
    """
    Admin Analytics & Dashboard Metrics REST Endpoint.
    """
    permission_classes = [permissions.IsAdminUser]

    def get(self, request):
        total_docs = Document.objects.count()
        total_chunks = sum(d.total_chunks for d in Document.objects.all())
        total_queries = AnalyticsMetric.objects.filter(event_type='query').count()
        avg_latency = AnalyticsMetric.objects.filter(event_type='query').aggregate(Avg('latency_ms'))['latency_ms__avg'] or 0
        total_tokens = AnalyticsMetric.objects.aggregate(Sum('tokens_used'))['tokens_used__sum'] or 0
        ocr_count = Document.objects.filter(is_ocr_processed=True).count()

        return Response({
            'total_documents': total_docs,
            'total_chunks': total_chunks,
            'total_queries': total_queries,
            'avg_latency_ms': round(avg_latency, 2),
            'estimated_tokens_used': total_tokens,
            'ocr_processed_docs': ocr_count,
            'system_status': 'Operational',
            'database_engine': getattr(settings, 'DB_ENGINE', 'sqlite')
        })
