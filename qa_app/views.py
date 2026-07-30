import json
import logging
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from .models import Document, Conversation, ChatMessage
from .utils import (
    process_document,
    generate_document_summary,
    retrieve_relevant_chunks,
    generate_answer,
    generate_quiz_questions,
    generate_cheat_sheet
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# AUTH VIEWS
# ─────────────────────────────────────────

def landing_page(request):
    """DocuChat AI Landing Page"""
    if request.user.is_authenticated:
        return redirect('documents')
    return render(request, 'landing.html')


def register_view(request):
    """User Registration View"""
    if request.user.is_authenticated:
        return redirect('documents')

    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f'Welcome {user.username}! Start by uploading a document.')
            return redirect('documents')
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, error)
    else:
        form = UserCreationForm()

    return render(request, 'register.html', {'form': form})


def login_view(request):
    """User Login View"""
    if request.user.is_authenticated:
        return redirect('documents')

    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, f'Welcome back, {user.username}!')
            return redirect('documents')
        else:
            messages.error(request, 'Invalid username or password.')
    else:
        form = AuthenticationForm()

    return render(request, 'login.html', {'form': form})


def logout_view(request):
    """User Logout View"""
    logout(request)
    messages.info(request, 'Logged out successfully.')
    return redirect('landing')


# ─────────────────────────────────────────
# DOCUMENT VIEWS
# ─────────────────────────────────────────

@login_required
def documents_view(request):
    """List uploaded documents for current user."""
    documents = Document.objects.filter(user=request.user)
    return render(request, 'documents.html', {
        'documents': documents,
        'total': documents.count()
    })


@login_required
def upload_document(request):
    """PDF upload karo aur RAG ingestion pipeline run karo."""
    if request.method != 'POST':
        return redirect('documents')

    title = request.POST.get('title', '').strip()
    pdf_file = request.FILES.get('pdf_file')

    errors = []
    if not title:
        errors.append('Please provide a document title.')
    elif len(title) > 200:
        errors.append('Title is too long (max 200 characters).')

    if not pdf_file:
        errors.append('Please select a PDF file.')
    elif not pdf_file.name.lower().endswith('.pdf'):
        errors.append('Only PDF files are accepted.')
    elif pdf_file.size > 10 * 1024 * 1024:  # 10MB
        errors.append('File too large. Maximum: 10MB.')

    if errors:
        for error in errors:
            messages.error(request, error)
        return redirect('documents')

    document = Document.objects.create(
        user=request.user,
        title=title,
        pdf_file=pdf_file,
        file_size=pdf_file.size,
    )

    document.pdf_file.open('rb')
    success = process_document(document, document.pdf_file)
    document.pdf_file.close()

    if success:
        try:
            document.summary = generate_document_summary(document.full_text)
            document.save()
        except Exception:
            pass

        messages.success(
            request,
            f'✅ "{title}" processed successfully! '
            f'{document.total_chunks} chunks created. Ready to chat & generate quiz!'
        )
    else:
        document.delete()
        messages.error(
            request,
            'Could not process the PDF. Make sure it is a text-based PDF.'
        )

    return redirect('documents')


@login_required
def delete_document(request, pk):
    """Delete uploaded document — cascades to chunks, conversations, messages."""
    doc = get_object_or_404(Document, pk=pk, user=request.user)
    doc.delete()
    messages.success(request, 'Document deleted successfully.')
    return redirect('documents')


# ─────────────────────────────────────────
# CHAT VIEWS
# ─────────────────────────────────────────

@login_required
def chat_view(request, doc_id):
    """Document ke saath chat interface."""
    document = get_object_or_404(Document, pk=doc_id, user=request.user)

    if request.method == 'GET':
        conversation, created = Conversation.objects.get_or_create(
            user=request.user,
            document=document,
            defaults={'title': f"Chat about {document.title[:30]}"}
        )

        all_conversations = Conversation.objects.filter(
            user=request.user,
            document=document,
        )

        messages_list = conversation.messages.all()

        return render(request, 'chat.html', {
            'document': document,
            'conversation': conversation,
            'conversations': all_conversations,
            'messages_list': messages_list,
            'active_conv_id': conversation.id,
        })

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            question = data.get('question', '').strip()
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid request'}, status=400)

        if not question:
            return JsonResponse({'error': 'Question cannot be empty'}, status=400)

        conversation, _ = Conversation.objects.get_or_create(
            user=request.user,
            document=document,
            defaults={'title': f"Chat about {document.title[:30]}"}
        )

        user_message = ChatMessage.objects.create(
            conversation=conversation,
            role='user',
            content=question,
        )

        history = [
            {'role': m.role, 'content': m.content}
            for m in conversation.messages.exclude(pk=user_message.pk).order_by('-timestamp')[:8]
        ]
        history.reverse()

        relevant_chunks = retrieve_relevant_chunks(document, question, conversation_history=history, top_k=4)

        try:
            answer = generate_answer(question, relevant_chunks, history)
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            answer = "⚠️ Could not generate an answer. Please try rephrasing your question."

        ai_message = ChatMessage.objects.create(
            conversation=conversation,
            role='assistant',
            content=answer,
            retrieved_chunks=relevant_chunks[:2],
        )

        conversation.save()

        return JsonResponse({
            'answer': answer,
            'chunks_used': len(relevant_chunks),
            'message_id': ai_message.pk,
        })

    return JsonResponse({'error': 'Method not allowed'}, status=405)


@login_required
def new_conversation(request, doc_id):
    """Naya conversation start karo same document pe."""
    document = get_object_or_404(Document, pk=doc_id, user=request.user)

    count = Conversation.objects.filter(
        user=request.user,
        document=document
    ).count()

    conversation = Conversation.objects.create(
        user=request.user,
        document=document,
        title=f"Chat #{count + 1} — {document.title[:30]}"
    )

    messages.success(request, 'New conversation started!')
    return redirect('chat_conversation', doc_id=document.id, conv_id=conversation.id)


@login_required
def chat_conversation_view(request, doc_id, conv_id):
    """Specific conversation ke saath chat karo."""
    document = get_object_or_404(Document, pk=doc_id, user=request.user)
    conversation = get_object_or_404(
        Conversation, pk=conv_id, user=request.user, document=document
    )

    if request.method == 'GET':
        all_conversations = Conversation.objects.filter(
            user=request.user,
            document=document,
        )

        messages_list = conversation.messages.all()

        return render(request, 'chat.html', {
            'document': document,
            'conversation': conversation,
            'conversations': all_conversations,
            'messages_list': messages_list,
            'active_conv_id': conversation.id,
        })

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            question = data.get('question', '').strip()
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid request'}, status=400)

        if not question:
            return JsonResponse({'error': 'Question cannot be empty'}, status=400)

        user_msg = ChatMessage.objects.create(
            conversation=conversation,
            role='user',
            content=question,
        )

        history = [
            {'role': m.role, 'content': m.content}
            for m in conversation.messages.exclude(pk=user_msg.pk).order_by('-timestamp')[:8]
        ]
        history.reverse()

        relevant_chunks = retrieve_relevant_chunks(document, question, conversation_history=history, top_k=4)

        try:
            answer = generate_answer(question, relevant_chunks, history)
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            answer = "⚠️ Could not generate an answer. Please try rephrasing."

        ai_msg = ChatMessage.objects.create(
            conversation=conversation,
            role='assistant',
            content=answer,
            retrieved_chunks=relevant_chunks[:2],
        )

        conversation.save()

        return JsonResponse({
            'answer': answer,
            'chunks_used': len(relevant_chunks),
            'message_id': ai_msg.pk,
        })

    return JsonResponse({'error': 'Method not allowed'}, status=405)


@login_required
def delete_conversation(request, doc_id, conv_id):
    """Conversation delete karo."""
    conversation = get_object_or_404(
        Conversation, pk=conv_id, user=request.user, document_id=doc_id
    )
    conversation.delete()
    messages.success(request, 'Conversation deleted.')
    return redirect('chat', doc_id=doc_id)


@login_required
def rename_conversation(request, doc_id, conv_id):
    """Conversation rename karo (AJAX)."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    try:
        data = json.loads(request.body)
        new_title = data.get('title', '').strip()
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid request'}, status=400)

    if not new_title or len(new_title) > 200:
        return JsonResponse({'error': 'Invalid title'}, status=400)

    conversation = get_object_or_404(
        Conversation, pk=conv_id, user=request.user, document_id=doc_id
    )
    conversation.title = new_title
    conversation.save()

    return JsonResponse({'success': True, 'title': new_title})


# ─────────────────────────────────────────
# ACTIVE LEARNING & ANALYTICS VIEWS
# ─────────────────────────────────────────

@login_required
def generate_quiz_view(request, doc_id):
    """Returns 5 MCQs generated from document content."""
    document = get_object_or_404(Document, pk=doc_id, user=request.user)
    quiz_data = generate_quiz_questions(document)
    return JsonResponse({'questions': quiz_data, 'title': document.title})


@login_required
def cheat_sheet_view(request, doc_id):
    """Renders print-ready Revision Cheat Sheet & Study Notes page."""
    document = get_object_or_404(Document, pk=doc_id, user=request.user)
    sheet_data = generate_cheat_sheet(document)
    return render(request, 'cheat_sheet.html', {
        'document': document,
        'sheet': sheet_data
    })


@login_required
def analytics_dashboard_view(request):
    """Renders Admin System & Usage Analytics Dashboard page."""
    return render(request, 'admin_analytics.html')


@login_required
def system_design_view(request):
    """Renders System Design & Architecture documentation page."""
    return render(request, 'system_design.html')
