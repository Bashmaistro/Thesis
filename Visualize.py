from tiatoolbox.wsicore.wsireader import WSIReader

wsi = WSIReader.open("tcga/Tcga_lgg_astro_ready/1fe1a0ed-c832-4096-a7cc-72b61d4fb592.dcm")

# Full resolution görüntüyü oku
full_image = wsi.read_region(location=(0, 0), size=wsi.level_dimensions[0], resolution=0.25, units="mpp")

full_image.show()  # PIL Image olarak
