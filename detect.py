import colour_recognition
import character_recognition
from pathlib import Path
from config import Settings
from saving import Saving
from utils.dataloader import LoadImages, LoadStreams, IMG_FORMATS, VID_FORMATS
from utils.general import LOGGER, Profile, colorstr
from detection_methods import Detection
from utils.results_storing import Storage


"""
The following code contains the detection of the square target and saves only the inner square data
"""

    
def run():
    # intialising the key information
    filename = 1
    config = Settings()
    
    source = str(config.source)
    webcam = config.source.isnumeric() or config.source.endswith('.txt') or config.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')) or config.source.endswith('.mp4')
    save_img = config.save_results and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    save = Saving(config.name_of_folder, config.exist_ok, save_img)
    store_results = Storage(config, save.save_dir, config.headings, config.title)
    store_speed = Storage(config, save.save_dir, ["Filename", "Detection Speed (ms)", "Character Speed (ms)", "Colour Speed (ms)", "Total Speed (ms)"], "Speed")
    detect = Detection(config)

    # Dataloader
    if webcam:
        # view_img = check_imshow()
        dataset = LoadStreams(source, img_size=config.model_input_size)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=config.model_input_size)
        bs = 1  # batch_size
    # vid_path, vid_writer = [None] * bs, [None] * bs

    # Running Inference
    seen, dt = 0, (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        predicted_character_list = []
        predicted_color_list = []
        if not webcam:
            filename = Path(path).stem
        # detection
        with dt[0]:
            detect.next_frame(im0s)

        current_detection_time = dt[0].dt * 1E3
        print(f"{current_detection_time}ms")
        
        
        # Character Recognition
        with dt[1]:
            for i in range(int(len(detect.storing_inner_boxes_data)/8)):  
                if config.recognition_method:
                    if detect.storing_inner_boxes_data[7+(8*i)]: 
                        seen += 1
                        predicted_character, contour_image, chosen_image = character_recognition.character(detect.storing_inner_boxes_data[0+(8*i)])
                        predicted_character_list.append(predicted_character)
                    else:
                        contour_image, chosen_image, predicted_character = None, None, None
                else:
                    contour_image, chosen_image, predicted_character = None, None, None       
        
        # Colour Recognition
        with dt[2]:
            for i in range(int(len(detect.storing_inner_boxes_data)/8)):  
                if config.recognition_method:
                    if detect.storing_inner_boxes_data[7+(8*i)]: 
                        predicted_color, processed_image = colour_recognition.colour(detect.storing_inner_boxes_data[0+(8*i)])
                        predicted_color_list.append(predicted_color)
                    else:
                        processed_image, predicted_color = None, None
                else:
                    processed_image, predicted_color = None, None     

        # save img
        if save_img:
            for i in range(int(len(detect.storing_inner_boxes_data)/8)):  
                name_of_results = ["color", "frame", "roi", "contour_image","processed_image", "chosen_image", "outer_edge", "inner_edge", "possible_target", "before_inner_edge_search"]
                image_results = [detect.storing_inner_boxes_data[0+(8*i)], detect.storing_inner_boxes_data[1+(8*i)], detect.storing_inner_boxes_data[2+(8*i)], contour_image, processed_image, chosen_image, detect.storing_inner_boxes_data[3+(8*i)], detect.storing_inner_boxes_data[5+(8*i)], detect.storing_inner_boxes_data[6+(8*i)], detect.storing_inner_boxes_data[4+(8*i)]]
                for value, data in enumerate(name_of_results):
                    image_name = f"{filename}_{data}_{i}.jpg"
                    image = image_results[value]
                    if image is not None:
                        save.save_the_image(image_name, image)

        # saving csv results
        for i in range(len(predicted_color_list)):
            print(f"predicted character and colour = {predicted_character_list[i]} and {predicted_color_list[i]}")
            results = [str(filename), str(i), str(predicted_character_list[i]), str(predicted_color_list[i])]
            store_results.prediction_results(results)

        detect_speed = dt[0].dt * 1E3
        if len(predicted_character_list) > 0:
            char_speed = dt[1].dt * 1E3
            colour_speed = dt[2].dt * 1E3
        else:
            char_speed = 0
            colour_speed = 0 

        total_speed = detect_speed + char_speed + colour_speed
        store_speed.prediction_results([str(filename), str(detect_speed), str(char_speed), str(colour_speed), str(total_speed)])

        print("Target Captured and saved to file")

    if seen != 0: 
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        total_speed = sum(t)

    text = f'Speed: %.1fms detection, %.1fms character recognition, %.1fms colour recognition per image at shape {(1, 3, (config.model_input_size, config.model_input_size))} total speed: {total_speed}' % t
    LOGGER.info(text)
    store_results.write_in_txt(str(text))
    LOGGER.info(f"Results saved to {colorstr('bold', save.save_dir)}{s}")


def main():
    print('Starting detection')
    run()


if __name__ == "__main__":
    main()
