from django.contrib import admin
from .models import Document, DocumentChunk, Conversation, ChatMessage


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ['title', 'user', 'total_chunks', 'file_size_display_admin', 'uploaded_at']
    list_filter = ['user', 'uploaded_at']
    search_fields = ['title', 'user__username']
    readonly_fields = ['full_text', 'total_chunks', 'file_size', 'uploaded_at']
    date_hierarchy = 'uploaded_at'

    def file_size_display_admin(self, obj):
        return obj.file_size_display
    file_size_display_admin.short_description = 'Size'


@admin.register(DocumentChunk)
class DocumentChunkAdmin(admin.ModelAdmin):
    list_display = ['document', 'chunk_index', 'word_count']
    list_filter = ['document']
    search_fields = ['chunk_text']
    readonly_fields = ['tfidf_vector']


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ['title', 'user', 'document', 'message_count_admin', 'created_at', 'updated_at']
    list_filter = ['user', 'document', 'created_at']
    search_fields = ['title', 'user__username']

    def message_count_admin(self, obj):
        return obj.message_count
    message_count_admin.short_description = 'Messages'


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ['conversation', 'role', 'content_preview', 'timestamp']
    list_filter = ['role', 'timestamp']
    search_fields = ['content']
    readonly_fields = ['retrieved_chunks']

    def content_preview(self, obj):
        return obj.content[:80] + "..." if len(obj.content) > 80 else obj.content
    content_preview.short_description = 'Content'
