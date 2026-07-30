from django.db import models
from django.contrib.auth.models import User
import json


class Document(models.Model):
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='documents'
    )
    title = models.CharField(max_length=200)
    pdf_file = models.FileField(upload_to='documents/')
    full_text = models.TextField(blank=True, default='')
    summary = models.TextField(blank=True, default='')
    total_chunks = models.IntegerField(default=0)
    file_size = models.IntegerField(default=0)  # bytes
    is_ocr_processed = models.BooleanField(default=False)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"{self.user.username} — {self.title}"

    @property
    def file_size_display(self):
        size = self.file_size
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size/1024:.1f} KB"
        else:
            return f"{size/1024/1024:.1f} MB"


class DocumentChunk(models.Model):
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name='chunks'
    )
    chunk_index = models.IntegerField()
    chunk_text = models.TextField()
    tfidf_vector = models.JSONField(default=dict)
    word_count = models.IntegerField(default=0)

    class Meta:
        ordering = ['document', 'chunk_index']

    def __str__(self):
        return f"{self.document.title} — Chunk {self.chunk_index}"


class Conversation(models.Model):
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='conversations'
    )
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name='conversations'
    )
    title = models.CharField(
        max_length=200,
        default='New Conversation'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']

    def __str__(self):
        return f"{self.user.username} × {self.document.title}"

    @property
    def message_count(self):
        return self.messages.count()


class ChatMessage(models.Model):
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
    ]

    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.CASCADE,
        related_name='messages'
    )
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    retrieved_chunks = models.JSONField(default=list)
    latency_ms = models.IntegerField(default=0)
    provider_used = models.CharField(max_length=50, default='Qwen3/Gemini')
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['timestamp']

    def __str__(self):
        return f"[{self.role}] {self.content[:50]}..."


class AnalyticsMetric(models.Model):
    """
    Analytics & Dashboard metrics model.
    Tracks API usage, token estimates, queries count, and OCR processes.
    """
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    event_type = models.CharField(max_length=50)  # 'query', 'ocr', 'vector_index', 'pdf_upload'
    tokens_used = models.IntegerField(default=0)
    latency_ms = models.IntegerField(default=0)
    status = models.CharField(max_length=20, default='success')  # 'success', 'error', 'rate_limited'
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"[{self.event_type}] {self.status} @ {self.timestamp.strftime('%Y-%m-%d %H:%M')}"
