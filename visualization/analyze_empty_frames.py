import os
import cv2
import pandas as pd

# write the function to analyze the empty frames csv 

def analyze_empty_frames(df):
    #frame_idx,frame_name,timestamp,num_detections,is_empty,is_bad_frame
    

    total_frames = len(df)
    bad_frames = df['is_bad_frame'].sum()
    empty_frames = df['is_empty'].sum()
    non_empty_frames = total_frames - empty_frames - bad_frames

    print(f"Total frames processed: {total_frames}")
    print(f"Bad frames: {bad_frames} ({(bad_frames/total_frames)*100:.2f}%)")
    print(f"Empty frames: {empty_frames} ({(empty_frames/total_frames)*100:.2f}%)")
    print(f"Non-empty frames: {non_empty_frames} ({(non_empty_frames/total_frames)*100:.2f}%)")


    ### frame_name: 20240905_064719_000

    ### 4 hours video in total 
    total_minutes = 4 * 60 * 60
    print(f"Total video duration (approx): {total_minutes:.2f} minutes")    
    ### empty_frames = empty_minutes
    empty_minutes = empty_frames
    print(f"Empty video duration (approx): {empty_minutes:.2f} minutes")
    non_empty_minutes = non_empty_frames
    print(f"Non-empty video duration (approx): {non_empty_minutes:.2f} minutes")

    ## calculate
    ratio = empty_minutes / total_minutes
    print(f"Ratio of empty video duration to total video duration: {ratio*100:.2f}%")




if __name__ == "__main__":

    file_dir = '/media/mu/zoo_vision/reports/'
    file = '/media/mu/zoo_vision/reports/video_empty_frames.csv'
    df = pd.read_csv(file)

    ### after 20240905-104719

    print("----- 02 - 06 AM -----")
    file=f'{file_dir}/ZAG-ELP-CAM-016-20240905-024719-1725497239539-7.csv'
    df = pd.read_csv(file)
    analyze_empty_frames(df)
    
    print("----- 14 - 18 PM -----")

    file=f'{file_dir}/ZAG-ELP-CAM-016-20240905-144719-1725540439722-7.csv'
    df = pd.read_csv(file)
    analyze_empty_frames(df)
    
    print("----- 18 - 22 PM -----")


    file=f'{file_dir}/ZAG-ELP-CAM-016-20240905-184718-1725554838999-7.csv'
    df = pd.read_csv(file)
    analyze_empty_frames(df)

    print("----- 22 - 02 AM -----")
    file=f'{file_dir}/ZAG-ELP-CAM-016-20240905-224718-1725569238475-7.csv'
    df = pd.read_csv(file)
    analyze_empty_frames(df)