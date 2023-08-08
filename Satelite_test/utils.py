"""
Utilities used by example notebooks
"""
import pandas as pd
import numpy as np
import math
import geopy
import geopy.distance
import matplotlib.pyplot as plt
from typing import Any, Optional, Tuple
import requests
from sentinelhub import (SHConfig, DataCollection, SentinelHubCatalog, SentinelHubRequest, BBox, bbox_to_dimensions, CRS, MimeType, Geometry)

def dms_to_decimal(input_deg):
    parts = input_deg.strip().split('°')
    degrees = int(float(parts[0]))
    min_part = parts[1].split('\'')
    minutes = int(float(min_part[0]))
    sec_part = min_part[1].split('"')
    seconds = float(sec_part[0])
    direction = sec_part[1]

    decimal_degrees = round(degrees + minutes / 60 + seconds / 3600, 8)

    if direction in ['S', 's', 'W', 'w']:
        decimal_degrees *= -1

    return decimal_degrees

def convert_dms_to_decimal(latitude_str, longitude_str):
    latitude_decimal = dms_to_decimal(latitude_str)
    longitude_decimal = dms_to_decimal(longitude_str)

    return latitude_decimal, longitude_decimal

def pretty_print_point(point, return_var=False):

    def format_coordinate(coordinate, direction):
        degrees = int(coordinate)
        minutes_float = (coordinate - degrees) * 60
        minutes = int(minutes_float)
        seconds = (minutes_float - minutes) * 60
        return f"{degrees}° {minutes}' {seconds:.2f}\" {direction}"

    latitude_direction = "N" if point.latitude >= 0 else "S"
    longitude_direction = "E" if point.longitude >= 0 else "W"

    latitude_str = format_coordinate(abs(point.latitude), latitude_direction)
    longitude_str = format_coordinate(abs(point.longitude), longitude_direction)

    if return_var:
        return f"{latitude_str}, {longitude_str}" 
    
    print(f"{latitude_str}, {longitude_str}")

def returnLeftBottomBoxCoord(long, lat, dist = 1):
    if math.isnan(float(long)) or math.isnan(float(lat)):
        return float('NaN')  
    start = geopy.Point(longitude=long, latitude=lat)
    d = geopy.distance.distance(kilometers = dist)
    point_south = d.destination(point=start, bearing=180)
    final_p = d.destination(point=point_south, bearing=270)
    return final_p.longitude, final_p.latitude

def returnRightTopBoxCoord(long, lat, dist = 1):
    if math.isnan(float(long)) or math.isnan(float(lat)):
        return float('NaN')
    start = geopy.Point(longitude=long, latitude=lat)
    d = geopy.distance.distance(kilometers = dist)
    point_north = d.destination(point=start, bearing=0)
    final_p = d.destination(point=point_north, bearing=90)
    return final_p.longitude, final_p.latitude

def plot_image(
    image: np.ndarray,
    factor: float = 1.0,
    clip_range: Optional[Tuple[float, float]] = None,
    **kwargs: Any
) -> None:
    """Utility function for plotting RGB images."""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])

def get_keycloak(username: str, password: str) -> str:
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
        }
    try:
        r = requests.post("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data=data,
        )
        r.raise_for_status()
    except Exception as e:
        raise Exception(
            f"Keycloak token creation failed. Reponse from the server was: {r.json()}"
            )
    return r.json()["access_token"]

def setup_sentinel_image_retrieval():
    start_date = "2022-06-01"
    end_date = "2022-06-10"
    data_collection = "SENTINEL-2"
    aoi = "POLYGON((4.220581 50.958859,4.521264 50.953236,4.545977 50.906064,4.541858 50.802029,4.489685 50.763825,4.23843 50.767734,4.192435 50.806369,4.189689 50.907363,4.220581 50.958859))'"
    json = requests.get(f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq '{data_collection}' and OData.CSC.Intersects(area=geography'SRID=4326;{aoi}) and ContentDate/Start gt {start_date}T00:00:00.000Z and ContentDate/Start lt {end_date}T00:00:00.000Z").json()
    # pd.DataFrame.from_dict(json['value']).head(5)   
    keycloak_token = get_keycloak("gonzalezsaezdiez@gmail.com", "Workwork2023!")
    print("keycloack_token done")
    session = requests.Session()
    session.headers.update({'Authorization': f'Bearer {keycloak_token}'})
    url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products(acdd7b9a-a5d4-5d10-9ac8-554623b8a0c9)/$value"
    response = session.get(url, allow_redirects=False)
    while response.status_code in (301, 302, 303, 307):
        url = response.headers['Location']
        response = session.get(url, allow_redirects=False)

    file = session.get(url, verify=False, allow_redirects=True)

    with open(f"product.zip", 'wb') as p:
        p.write(file.content)

def get_image_from_sentinel(aoi_coords_wgs84, start_date, end_date, eval_script, resolution: int = 10):
    config = SHConfig()

    config.sh_client_id = "sh-6b51cffc-a250-4249-8907-4fc56ae1a689 "
    config.sh_client_secret = "ZHVi8DQGf8vioS1VjAdjcbt2QobzE89a"
    config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"
    config.save("cdse")
    # Saved config can be later accessed with config = SHConfig("cdse")

    config = SHConfig("cdse")

    aoi_bbox = BBox(bbox=aoi_coords_wgs84, crs=CRS.WGS84)
    aoi_size = bbox_to_dimensions(aoi_bbox, resolution=resolution)

    catalog = SentinelHubCatalog(config=config)

    aoi_bbox = BBox(bbox=aoi_coords_wgs84, crs=CRS.WGS84)
    #time_interval = '2022-01-01', '2023-07-20'

    #search_iterator = catalog.search(
    #    DataCollection.SENTINEL2_L2A,
    #    bbox=aoi_bbox,
    #    time=time_interval,
    #    fields={"include": ["id", "properties.datetime"], "exclude": []},
    #
    # )

    #results = list(search_iterator)

    request_true_color = SentinelHubRequest(
    evalscript=eval_script,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A.define_from(
                name="s2", service_url="https://sh.dataspace.copernicus.eu"
            ),
            time_interval=(start_date, end_date),
            other_args={"dataFilter": {"mosaickingOrder": "leastCC"}}           )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
    bbox=aoi_bbox,
    size=aoi_size,
    config=config,
    )

    true_color_imgs = request_true_color.get_data()

    image = true_color_imgs[0]

    return image