import flask
from flask import Flask,request,render_template


app=Flask(__name__)
@app.route("/")
def index():
    return{"status:0"}
if __name__=="__main__":
    app.run(post=8085)
