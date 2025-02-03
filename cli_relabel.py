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

    # Get classified files
    list_pnp = sorted(glob.glob(os.path.join(output_folder, "*_2class_*.tif")))
    list_3commodity = sorted(glob.glob(os.path.join(output_folder, "*_3class_*.tif")))

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
