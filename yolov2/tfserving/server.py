#!/usr/bin/env python
# This script is used to start/stop TF Serving Server
import os
import signal
import subprocess

ACTIVATION_COMMAND = "tensorflow_model_server --port={} --model_name={} --model_base_path={}"


class ObjectDetectionServer(object):
    def __init__(self, model, model_path, port=9000):
        self.server     = None
        self.is_running = False
        self.model_path = model_path
        self.model      = model
        self.port       = port

    def start(self):
        return self._callback(command='start')

    def stop(self):
        return self._callback(command='stop')

    def _callback(self, command):
        if command == 'start':
            if not self.is_running:
                self.is_running = True
                print("Tensorflow Serving Server is launching at ")
                self.server = subprocess.Popen(ACTIVATION_COMMAND.format(self.port,
                                                                         self.model,
                                                                         self.model_path),
                                               stdin=subprocess.PIPE, shell=True)
                print("TF Serving Server is started at PID %s\n" % self.server.pid)
            else:
                print("Tensorflow Serving Server has been activated already..\n")

        if command == 'stop':
            if self.is_running:
                self.is_running = True
                self._turn_off_server()

                print("Tensorflow Serving Server is off..\n")
            else:
                print("Tensorflow Serving Server is not activated yet..\n")

    def _turn_off_server(self):
        ps_command = subprocess.Popen("ps -o pid --ppid %d --noheaders" % self.server.pid,
                                      shell=True,
                                      stdout=subprocess.PIPE)

        ps_output = ps_command.stdout.read()
        return_code = ps_command.wait()
        for pid_str in ps_output.split("\n")[:-1]:
            os.kill(int(pid_str), signal.SIGINT)
        self.server.terminate()