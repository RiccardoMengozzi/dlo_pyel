import os, shutil


input_folers = [
    "../DATA/action_5cm_22deg/1",
    "../DATA/action_5cm_22deg/2",
    "../DATA/action_5cm_22deg/3",
    "../DATA/action_5cm_22deg/4",
    "../DATA/action_5cm_22deg/5",
    "../DATA/action_5cm_22deg/6",
    "../DATA/action_5cm_22deg/7",
    "../DATA/action_5cm_22deg/8",
    "../DATA/action_5cm_22deg/9",
    "../DATA/action_5cm_22deg/10",
]


output_path = "new_dataset"


os.makedirs(output_path, exist_ok=True)

counter = 0

for it, input_foler in enumerate(input_folers):

    files = os.listdir(input_foler)
    files.sort()

    for file in files:
        file_name = file.split(".")[0]
        out_name = str(counter).zfill(4) + "_" + file_name.split("_")[1] + ".pkl"
        shutil.copy(os.path.join(input_foler, file), os.path.join(output_path, out_name))

        counter += 1
