from rest_framework import serializers
from django.contrib.auth.models import User
from .models import Document, DocumentChunk, Conversation, ChatMessage, AnalyticsMetric


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'date_joined']


class DocumentChunkSerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentChunk
        fields = ['id', 'chunk_index', 'chunk_text', 'word_count']


class DocumentSerializer(serializers.ModelSerializer):
    user = serializers.StringRelatedField(read_only=True)
    file_size_display = serializers.ReadOnlyField()

    class Meta:
        model = Document
        fields = [
            'id', 'user', 'title', 'pdf_file', 'summary',
            'total_chunks', 'file_size', 'file_size_display',
            'is_ocr_processed', 'uploaded_at'
        ]


class ChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = [
            'id', 'conversation', 'role', 'content',
            'retrieved_chunks', 'latency_ms', 'provider_used', 'timestamp'
        ]


class ConversationSerializer(serializers.ModelSerializer):
    user = serializers.StringRelatedField(read_only=True)
    document_title = serializers.ReadOnlyField(source='document.title')
    message_count = serializers.ReadOnlyField()

    class Meta:
        model = Conversation
        fields = [
            'id', 'user', 'document', 'document_title',
            'title', 'message_count', 'created_at', 'updated_at'
        ]


class AnalyticsMetricSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalyticsMetric
        fields = '__all__'
