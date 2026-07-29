from django.db import models
from django.contrib.auth.models import User
import json


class Document(models.Model):
    """
    User ka uploaded document store karta hai.

    ER Diagram:
    USER ──< DOCUMENT ──< DOCUMENT_CHUNK
    (one user → many docs → many chunks per doc)
    """
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='documents'
    )
    title = models.CharField(max_length=200)
    pdf_file = models.FileField(upload_to='documents/')
    full_text = models.TextField(blank=True, default='')
    summary = models.TextField(blank=True, default='')  # Auto-generated summary
    total_chunks = models.IntegerField(default=0)
    file_size = models.IntegerField(default=0)  # bytes
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"{self.user.username} — {self.title}"

    @property
    def file_size_display(self):
        """Human-readable file size"""
        size = self.file_size
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size/1024:.1f} KB"
        else:
            return f"{size/1024/1024:.1f} MB"


class DocumentChunk(models.Model):
    """
    Document ko chhote chunks mein store karta hai — RAG ke liye.

    Har chunk:
    - 500 words ka text
    - Us text ka TF-IDF vector (JSON mein store)

    Retrieval pe:
    - Question vectorize karo
    - Sab chunks ke saath cosine similarity calculate karo
    - Top 3-5 chunks return karo
    """
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name='chunks'
    )
    chunk_index = models.IntegerField()      # Chunk number (0, 1, 2...)
    chunk_text = models.TextField()          # Actual text content

    # TF-IDF vector JSON mein store karenge
    # Example: {"python": 0.8, "django": 0.6, "api": 0.4}
    tfidf_vector = models.JSONField(default=dict)

    word_count = models.IntegerField(default=0)

    class Meta:
        ordering = ['document', 'chunk_index']

    def __str__(self):
        return f"{self.document.title} — Chunk {self.chunk_index}"


class Conversation(models.Model):
    """
    Ek user aur ek document ke beech ki conversation.

    USER ──< CONVERSATION ──< CHAT_MESSAGE
    USER ──< DOCUMENT
    DOCUMENT ──< CONVERSATION
    """
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
    """
    Har individual message — user ka question ya AI ka answer.

    Role:
    - 'user' → user ne likha
    - 'assistant' → Gemini AI ne jawab diya
    """
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

    # Agar role='assistant' hai toh kaunse chunks se answer aaya?
    retrieved_chunks = models.JSONField(default=list)

    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['timestamp']  # Oldest first (chat order)

    def __str__(self):
        return f"[{self.role}] {self.content[:50]}..."
