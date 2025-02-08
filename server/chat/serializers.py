from rest_framework import serializers

class ChatRequestSerializer(serializers.Serializer):
    prompt = serializers.CharField()