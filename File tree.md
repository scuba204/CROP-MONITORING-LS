# File Tree: CROP-MONITORING-LS

Generated on: 9/18/2025, 10:32:48 AM
Root path: `c:\Users\nkosi\Desktop\CROP-MONITORING-LS`

```
├── 📁 .git/ 🚫 (auto-hidden)
├── 📁 TESTS/
│   ├── 📁 __pycache__/ 🚫 (auto-hidden)
│   ├── 🐍 Model_Evaluation.py
│   ├── 🐍 TEST.py
│   ├── 🐍 TEST_GEE_FUNCTIONS.py
│   └── 🐍 test_ndvi.py
├── 📁 TRAINING/
│   ├── 🐍 __init__.py
│   ├── 🐍 crop_rotation_model1.py
│   ├── 🐍 crop_rotation_predict_crop.py
│   ├── 🌐 crop_weed_map.html
│   ├── 🌐 crop_weed_map_filtered.html
│   ├── 🌐 crop_weed_map_filtered_spaced.html
│   ├── 🐍 drought_model.py
│   ├── 🐍 generate_unlabelled_points.py
│   ├── 🐍 heat_model.py
│   ├── 🐍 irrigation_model.py
│   ├── 🐍 nutrient_model.py
│   ├── 🐍 predict_crop_weed.py
│   ├── 🐍 train_anomaly_detector.py
│   ├── 🐍 train_crop_classifier_model.py
│   ├── 🐍 train_disease_model.py
│   ├── 🐍 visualize_predictions.py
│   ├── 🐍 visualize_predictions_map.py
│   ├── 🐍 visualize_predictions_map_filtered.py
│   └── 🐍 visualize_predictions_map_filtered_spaced.py
├── 📁 TRAINING_DATA/
│   ├── 🐍 Diseases_Synthetic data.py
│   ├── 📄 Pest_Synthetic_Data
│   └── 🐍 generate_synthetic_data.py
├── 📁 __pycache__/ 🚫 (auto-hidden)
├── 📁 configs/
│   ├── ⚙️ config.yaml
│   └── ⚙️ config_disease.yaml
├── 📁 data/
│   ├── 📁 LSO_adm/
│   │   ├── 📁 Ecological Zones/
│   │   │   ├── 📄 Zones_Final.dbf
│   │   │   ├── 📄 Zones_Final.prj
│   │   │   ├── 📄 Zones_Final.sbn
│   │   │   ├── 📄 Zones_Final.sbx
│   │   │   ├── 📄 Zones_Final.shp
│   │   │   ├── 🔒 Zones_Final.shp.LITSIBA-PC.2340.4316.sr.lock
│   │   │   ├── 🔒 Zones_Final.shp.LITSIBA-PC.6952.3956.sr.lock
│   │   │   ├── 📄 Zones_Final.shp.xml
│   │   │   ├── 📄 Zones_Final.shx
│   │   │   ├── 📄 Zones_FinalCopy.dbf
│   │   │   ├── 📄 Zones_FinalCopy.prj
│   │   │   ├── 📄 Zones_FinalCopy.sbn
│   │   │   ├── 📄 Zones_FinalCopy.sbx
│   │   │   ├── 📄 Zones_FinalCopy.shp
│   │   │   ├── 📄 Zones_FinalCopy.shp.xml
│   │   │   └── 📄 Zones_FinalCopy.shx
│   │   ├── 📄 LSO_adm0.cpg
│   │   ├── 📄 LSO_adm0.csv
│   │   ├── 📄 LSO_adm0.dbf
│   │   ├── 📄 LSO_adm0.prj
│   │   ├── 📄 LSO_adm0.sbn
│   │   ├── 📄 LSO_adm0.sbx
│   │   ├── 📄 LSO_adm0.shp
│   │   ├── 🔒 LSO_adm0.shp.DESKTOP-57C6386.7856.7924.sr.lock
│   │   ├── 🔒 LSO_adm0.shp.DESKTOP-57C6386.8776.7924.sr.lock
│   │   ├── 📄 LSO_adm0.shx
│   │   ├── 📄 LSO_adm1.cpg
│   │   ├── 📄 LSO_adm1.csv
│   │   ├── 📄 LSO_adm1.dbf
│   │   ├── 📄 LSO_adm1.prj
│   │   ├── 📄 LSO_adm1.sbn
│   │   ├── 📄 LSO_adm1.sbx
│   │   ├── 📄 LSO_adm1.shp
│   │   ├── 🔒 LSO_adm1.shp.DESKTOP-57C6386.11396.11488.sr.lock
│   │   ├── 🔒 LSO_adm1.shp.DESKTOP-57C6386.7500.11488.sr.lock
│   │   ├── 📄 LSO_adm1.shp.xml
│   │   ├── 📄 LSO_adm1.shx
│   │   ├── 📄 Lesotho.kmz
│   │   └── 📜 license.txt
│   ├── 📄 FAOSTAT_data_en_8-19-2025.csv
│   ├── 📄 crop_disease_training_data.csv
│   ├── 📄 crop_weed_training.csv
│   ├── 📄 drought_training.csv
│   ├── 📄 faostat_lesotho_area_shares.csv
│   ├── 📄 faostat_lesotho_yield_only.csv
│   ├── 📄 faostat_rotation_proxy_features.csv
│   ├── 📄 features.csv
│   ├── 📄 gee_env_features_final.csv
│   ├── 📄 gee_soil_features.csv
│   ├── 📄 heat_training.csv
│   ├── 📄 irrigation_training.csv
│   ├── 📄 nutrient_training.csv
│   ├── 🌐 predictions_map.html
│   ├── 📄 predictions_output.csv
│   ├── 📄 rotation_training_clean.csv
│   ├── 📄 unlabelled_field_data.csv
│   └── 📄 yield_training.csv
├── 📁 models/
│   ├── 📄 anomaly_features.json
│   ├── 📄 classification_report.csv
│   ├── 📄 crop_anomaly_detector.pkl
│   ├── 📄 crop_classifier_features.json
│   ├── 📄 crop_classifier_model.pkl
│   ├── 📄 crop_rotation_metadata.json
│   ├── 📄 crop_rotation_pipeline.joblib
│   ├── 📄 crop_vs_weed_model.pkl
│   ├── 📄 disease_risk_features.json
│   ├── 📄 disease_risk_model.pkl
│   ├── 📄 feature_list.json
│   ├── 📄 irrigation_model_features_v1.0.0.json
│   ├── 📄 irrigation_optimizer_v1.0.0_20250915_204907.pkl
│   ├── 📄 pest_model_features.json
│   ├── 📄 pest_risk_model.pkl
│   └── 📄 shap_summary.csv
├── 📁 outputs/
│   ├── 📄 crop_weed_training.cpg
│   ├── 📄 crop_weed_training.csv
│   ├── 📄 crop_weed_training.dbf
│   ├── 📄 crop_weed_training.prj
│   ├── 📄 crop_weed_training.shp
│   ├── 📄 crop_weed_training.shx
│   ├── 📄 crop_weed_training_metadata.json
│   ├── 📄 disease_training.csv
│   ├── 📄 disease_training_metadata.json
│   ├── 📄 pest_training.csv
│   └── 📄 pest_training_metadata.json
├── 📁 scripts/
│   ├── 📁 __pycache__/ 🚫 (auto-hidden)
│   ├── 🐍 __init__.py
│   ├── 🐍 data_loader.py
│   ├── 🐍 extract_features.py
│   ├── 🐍 faostat_preprocessing.py
│   ├── 🐍 feature_engineering.py
│   ├── 🐍 gee_export_env_features.py
│   ├── 🐍 gee_functions.py
│   ├── 🐍 generate_irrigation_dataset.py
│   └── 🐍 merge_features.py
├── 🐍 GEE_AUTHENTICATION.py
├── 📖 README.md
├── 🐍 __init__.py
├── 🐍 daily farm monitor.py
├── 🐍 daily_crop_monitoring_dashboard_app.py
├── 📄 disease_risk_features.json
├── 🐍 palettes.py
├── 📄 requirements.txt
└── 📄 rotation_yield_pipeline.pkl
```

---
*Generated by FileTree Pro Extension*