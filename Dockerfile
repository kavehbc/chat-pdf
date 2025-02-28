# docker build --progress=plain --no-cache -t kavehbc/chat-pdf .
# docker save -o chat-pdf.tar kavehbc/chat-pdf
# docker load --input chat-pdf.tar

FROM python:3.9-buster

LABEL version="1.0.0"
LABEL maintainer="Kaveh Bakhtiyari"
LABEL url="http://bakhtiyari.com"
LABEL vcs-url="https://github.com/kavehbc/chat-pdf"
LABEL description="Chat with PDF files using Google Gemini"

WORKDIR /app
COPY . .

# installing the requirements
RUN pip install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]