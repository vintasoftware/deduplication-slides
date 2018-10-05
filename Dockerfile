FROM jupyter/scipy-notebook:1145fb1198b2

USER ${NB_USER}
COPY reduced-requirements.txt /tmp/
RUN pip install -r /tmp/reduced-requirements.txt
CMD ["jupyter", "notebook", "--ip", "0.0.0.0"]
