from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

# Define urlpatterns first
urlpatterns = [
    path('', views.index, name='index'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('stop_feed/', views.stop_feed, name='stop_feed'),
    path('predict/', views.predict_image, name='predict_image'),
]

# Append static file serving for development
if settings.DEBUG:  # Only serve static files in development
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
