import os
import glob
import joblib
import argparse
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from shapely.geometry import box

def main(input_folder, output_folder, lulc_path):
    path_raster = glob.glob(os.path.join(input_folder, "*.tif"))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load models
    model_2class = {'MLP': joblib.load('./Model_Generated/02_Paddy_Others/MLP_paddy_others_best_model.joblib')}
    model_3class = {'ExtraTrees': joblib.load('./Model_Generated/05_jagung_tebu_others/ExtraTrees_jagung_tebu_others_best_model.joblib')}

    for raster_path in path_raster:
        filename = os.path.basename(raster_path)

        # Read GeoTIFF
        with rasterio.open(raster_path) as src:
            raster_data = src.read()
            profile = src.profile

        num_bands, height, width = raster_data.shape

        # Check band count
        if num_bands != 181:
            print(f"Warning: Image {raster_path} does not have 181 bands. Skipping this image.")
            continue

        # Process with MLP 2-class
        flat_data = raster_data.reshape(num_bands, -1).T

        for model_name, model in model_2class.items():
            predictions = model.predict(flat_data)
            predicted_classes = predictions.reshape(height, width)
            output_file = os.path.join(output_folder, f"classified_2class_{filename}")

            with rasterio.open(
                output_file,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=rasterio.uint8,
                crs=src.crs,
                transform=src.transform,
            ) as dst:
                dst.write(predicted_classes.astype(np.uint8), 1)

            print(f"2 Class Predictions saved to {output_file}")

        for model_name, model in model_3class.items():
            predictions = model.predict(flat_data)
            predicted_classes = predictions.reshape(height, width)
            output_file = os.path.join(output_folder, f"classified_3class_{filename}")

            with rasterio.open(
                output_file,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=rasterio.uint8,
                crs=src.crs,
                transform=src.transform,
            ) as dst:
                dst.write(predicted_classes.astype(np.uint8), 1)

            print(f"3 Class Predictions saved to {output_file}")

    print(f"Processing complete. Classified images saved to {output_folder}")

    # Get classified files
    list_pnp = sorted(lob.glob(os.path.join(output_folder, "*2class*.tif")))
    list_3commodity = sorted(glob.glob(os.path.join(output_folder, "*3class*.tif")))

    print("list_pnp:", list_pnp)
    print("list_3commodity:", list_3commodity)

    # Label mapping
    label_mapping = {
        11: 1,  # Padi
        12: 1,  # Padi
        13: 1,  # Padi
        21: 2,  # Jagung
        22: 3,  # Tebu
        23: 4   # Others
    }

    def relabel(value):
        return label_mapping.get(value, 0)

    # Read and preprocess the LULC raster
    with rasterio.open(lulc_path) as lulc_ds:
        lulc_data = lulc_ds.read(1)
        lulc_transform = lulc_ds.transform
        lulc_crs = lulc_ds.crs

    # Process classified images
    for pnp_path, comm_path in zip(list_pnp, list_3commodity):
        with rasterio.open(pnp_path) as pnp_ds:
            pnp_data = pnp_ds.read(1)
            pnp_meta = pnp_ds.meta.copy()
            pnp_bounds = pnp_ds.bounds
            pnp_geometry = box(*pnp_bounds)
            pnp_geometry_geojson = [pnp_geometry.__geo_interface__]

        # Crop the LULC mask
        with rasterio.open(lulc_path) as lulc_ds:
            with WarpedVRT(
                    lulc_ds,
                    crs=pnp_ds.crs,
                    transform=pnp_ds.transform,
                    width=pnp_ds.width,
                    height=pnp_ds.height,
                    resampling=Resampling.nearest,
            ) as lulc_vrt:
                cropped_lulc, cropped_transform = mask(lulc_vrt, pnp_geometry_geojson, crop=True)
                cropped_lulc = cropped_lulc[0]
                cropped_lulc = np.where(cropped_lulc == 40, 1, np.nan)

        masked_pnp = np.where(cropped_lulc == 1, pnp_data, np.nan)

        with rasterio.open(comm_path) as comm_ds:
            comm_data = comm_ds.read(1)

        masked_comm = np.where(cropped_lulc == 1, comm_data, np.nan)

        final_output = (masked_pnp + 1) * 10 + (masked_comm + 1)
        relabeled_output = np.vectorize(relabel)(final_output)

        output_path = os.path.join(output_folder, f"commodity_final_relabel_{os.path.basename(pnp_path)}")
        pnp_meta.update({
            "driver": "GTiff",
            "dtype": "uint8",
            "nodata": 0,
            "height": relabeled_output.shape[0],
            "width": relabeled_output.shape[1],
            "transform": cropped_transform
        })

        with rasterio.open(output_path, "w", **pnp_meta) as dst:
            dst.write(relabeled_output.astype(np.uint8), 1)

        print(f"Saved relabeled raster to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify and process raster images.")
    parser.add_argument("input_folder", type=str, help="Path to the input folder containing GeoTIFF images.")
    parser.add_argument("output_folder", type=str, help="Path to the output folder for classified images.")
    parser.add_argument("lulc_path", type=str, help="Path to the LULC raster file.")

    args = parser.parse_args()
    main(args.input_folder, args.output_folder, args.lulc_path)
