import os
from flask import request, jsonify
import json
from sre import app, db, JSONEncoder
import subprocess
import hashlib
from base64 import b64decode
import logging
from sre.services.service_email import ServiceEmail

RELOAD_COMMAND = ' sudo /usr/sbin/nginx -s reload '
list_vms = []


@app.route('/reload_nginx', methods=['POST'], strict_slashes=False)
def reload_nginx_in_vm():
    jsonContent = {'Content-Type': 'application/json'}
    status = 0
    if request.method == "POST":
        content = request.get_json()
    else:
        return json.dumps({
            'success': "Error"})

    if content["is_vm_need_reload"] == True:
        app.logger.info('Ip real %s ', content["real_ip"])

        db.monitor.insert_one({
            # "rpaas-fe-cme": content["rpaas-fe-cme"],
            'vm_status': {
                'real_ip': content["real_ip"],
                'is_vm_need_reload': content["is_vm_need_reload"],
                'access-date': content["access_date"],
                'modify-date': content["modify_date"]
            }
        })

        service_email = ServiceEmail("disalles7@gmail.com",
                                     "disalles222@hotmail.com",
                                     "disalles7@gmail.com", "", "VM teste ", "O IP = "+content["real_ip"])
        service_email.sendEmail()

        return json.dumps({
            # "rpaas-fe-cme": content["rpaas-fe-cme"],
            'vm_status': {
                'real_ip': content["real_ip"],
                'is_vm_need_reload': content["is_vm_need_reload"],
                'access-date': content["access_date"],
                'modify-date': content["modify_date"]
            }
        }), 200, jsonContent
    else:
        return json.dumps({
            "Status": "Essa Vm Nao precisa de Reload"
        }), 200, jsonContent


@app.route("/healthcheck", methods=['GET'], strict_slashes=False)
def healthcheck():
    return jsonify({'status': 'online'})


@app.route("/list_vm_status", methods=['GET'], strict_slashes=False)
def get_vm_status():
    vms = list(db.monitor.find({}))
    return json.dumps(vms, cls=JSONEncoder), 200, {'Content-Type': 'application/json'}
