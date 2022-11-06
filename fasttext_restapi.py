from flask import Flask, request, make_response, jsonify, render_template, redirect, url_for
from flask_cors import CORS
from prediction import find, change_word, sentiment_predict
# from main import main
import json
import re
import os

app = Flask(__name__)
CORS(app)

# app.register_blueprint(main.demo_main, url_prefix="/main")


@app.route("/")
def main():
    return make_response("hi")


# @app.route("/main/<tmp>&&<origin>")
# def main2(tmp, origin):
#     return render_template("product.html", tmp=tmp, origin=origin)


# @app.route("/main/test")
# def main_test():
#     # 입력받은 문자열
#     origin = request.args.get("inputtext")
#     sentence = request.args.get("inputtext")
#     # 금지 단어
#     word = find(sentence)
#     score = sentiment_predict(sentence)
#     if word == None:
#         print("금지어가 포함되어 있지 않습니다")
#     else:
#         if score <= 0.25:
#             for i in word:
#                 for j in i:
#                     p2 = re.compile(j[0]+".+?"+j[-1])
#                     p = re.compile(j[0]+j[-1])

#                     li = sentence.split(' ')
#                     for a in li:
#                         result = p.match(a)
#                         result2 = p2.match(a)
#                         if result != None:
#                             sentence = sentence.replace(
#                                 str(result.group()), '***')
#                             break
#                         elif result2 != None:
#                             sentence = sentence.replace(
#                                 str(result2.group()), '***')
#                             break

#     return redirect(url_for('main2', tmp=sentence, origin=origin))


@app.route("/test/<comment>", methods=['GET', 'POST'])
def test(comment):
    if request.method == "GET":
        print('GET')
        li = find(comment)
        score = sentiment_predict(comment)
        if li == None:
            #print("금지어가 포함되지 않은 댓글입니다.")
            return make_response(json.dumps(li, ensure_ascii=False))
        else:
            li['score'] = score*100
            #print(json.dumps(li, ensure_ascii=False))
            return make_response(json.dumps(li, ensure_ascii=False))

    if request.method == "POST":
        print("POST")

    return make_response(jsonify({'status': True}), 200)


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port="8080")
