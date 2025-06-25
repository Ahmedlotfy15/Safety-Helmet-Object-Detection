from roboflow import Roboflow

rf = Roboflow(api_key="itkD6guvd0yinyQppFGF")
project = rf.workspace("ahmed-lotfi").project("safety-helmet-q3b8o-frkjj")
version = project.version(1)
dataset = version.download("yolov11")