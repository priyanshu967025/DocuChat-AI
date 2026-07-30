from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views, api_views

router = DefaultRouter()
router.register(r'api/documents', api_views.DocumentViewSet, basename='api_document')
router.register(r'api/conversations', api_views.ConversationViewSet, basename='api_conversation')

urlpatterns = [
    # HTML Views
    path('', views.landing_page, name='landing'),
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),

    # Documents
    path('documents/', views.documents_view, name='documents'),
    path('documents/upload/', views.upload_document, name='upload_document'),
    path('documents/<int:pk>/delete/', views.delete_document, name='delete_document'),
    path('documents/<int:doc_id>/cheatsheet/', views.cheat_sheet_view, name='cheat_sheet'),

    # Chat
    path('chat/<int:doc_id>/', views.chat_view, name='chat'),
    path('chat/<int:doc_id>/conversation/<int:conv_id>/',
         views.chat_conversation_view, name='chat_conversation'),

    # Active Learning & Analytics
    path('chat/<int:doc_id>/quiz/', views.generate_quiz_view, name='generate_quiz'),
    path('analytics/', views.analytics_dashboard_view, name='analytics_dashboard'),

    # Conversation management
    path('chat/<int:doc_id>/new/', views.new_conversation, name='new_conversation'),
    path('chat/<int:doc_id>/conversation/<int:conv_id>/delete/',
         views.delete_conversation, name='delete_conversation'),
    path('chat/<int:doc_id>/conversation/<int:conv_id>/rename/',
         views.rename_conversation, name='rename_conversation'),

    # REST API & DRF Endpoints
    path('', include(router.urls)),
    path('api/search/', api_views.SemanticSearchAPIView.as_view(), name='api_search'),
    path('api/analytics/', api_views.AnalyticsDashboardAPIView.as_view(), name='api_analytics'),
]
