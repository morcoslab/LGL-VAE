FROM python:3

COPY . .

RUN pip install -r requirements.txt
EXPOSE 5006
CMD [ "python", "vae_lgl_analysis.py"]