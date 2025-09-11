import ee

ee.Initialize(project='winged-tenure-464005-p9')



print(ee.ImageCollection("MODIS/061/MOD16A2").first().bandNames().getInfo())
