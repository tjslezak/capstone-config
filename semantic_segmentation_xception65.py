import os
import rastervision as rv


def build_scene(task, data_uri, id, channel_order=None):
    ## id = id.replace('-', '_')
    raster_source_uri = '{}/rasters/{}_raster.tif'.format(data_uri, id)
    label_source_uri = '{}/labels/{}_labels.tif'.format(data_uri, id)

    # Using with_rgb_class_map because input TIFFs have classes encoded as RGB colors.
    label_source = rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION_RASTER) \
        .with_rgb_class_map(task.class_map) \
        .with_raster_source(label_source_uri) \
        .build()

    # URI will be injected by scene config.
    # Using with_rgb(True) because we want prediction TIFFs to be in RGB format.
    label_store = rv.LabelStoreConfig.builder(rv.SEMANTIC_SEGMENTATION_RASTER) \
        .with_rgb(True) \
        .build()

    # Must define raster_source separate from raster_source_uri so StatsTransformer
    # can convert uint16 images to uint8
    raster_source = rv.RasterSourceConfig.builder(rv.GEOTIFF_SOURCE) \
        .with_uri(raster_source_uri) \
        .with_stats_transformer() \
        .build()

    scene = rv.SceneConfig.builder() \
                          .with_task(task) \
                          .with_id(id) \
                          .with_raster_source(raster_source, channel_order=channel_order) \
                          .with_label_source(label_source) \
                          .with_label_store(label_store) \
                          .build()

    return scene


class GeoSemanticSegmentation(rv.ExperimentSet):
    def exp_main(self, root_uri, data_uri, test_run=False):
        """Run an experiment on Sentinel-2 data of Arizona.
        Uses Tensorflow Deeplab backend with Mobilenet architecture.
        Args:
            root_uri: (str) root directory for experiment output
            data_uri: (str) root directory of data
            test_run: (bool) if True, run a very small experiment as a test and generate
                debug output
        """
        if test_run == 'True':
            test_run = True
        elif test_run == 'False':
            test_run = False

        train_ids = ['RTU', 'RUU', 'RUV', 'RVU', 'RVV', 'RWU', 'RWV', 'RXU', 'RXV', 'SQA',
                    'SQR', 'SQS', 'SQT', 'SQU', 'SQV', 'STA', 'STB', 'STC', 'STD', 'STE', 'STF',
                    'SUA', 'SUB', 'SUC', 'SUD', 'SUE', 'SUF', 'SVA', 'SVB', 'SVC', 'SVD', 'SVE',
                    'SVF', 'SWA', 'SWB', 'SWC', 'SWD', 'SWE', 'SWF', 'SXA', 'SXB', 'SXC', 'SXD',
                    ]
        val_ids = ['SXE', 'SXF', 'SYC', 'SYD']

        # blue, red, ir
        channel_order = [0, 1, 2]

        debug = False
        batch_size = 8
        chips_per_scene = 225
        num_steps = 150000
        model_type = rv.XCEPTION_65

        if test_run:
            debug = True
            num_steps = 1
            batch_size = 1
            chips_per_scene = 50
            train_ids = train_ids[0:1]
            val_ids = val_ids[0:1]

        classes = {'Q': (0, 'rgb(244,242,230)'), 'QTb': (1, 'rgb(240,213,240)'),
                    'QTv': (2, 'rgb(227,196,240)'), 'Qr': (3, 'rgb(255,255,186)'),
                     'Qy': (4, 'rgb(255,255,227)'), 'Qo': (5, 'rgb(253,252,155)'),
                     'QTs': (6, 'rgb(255,232,186)'), 'Tvy': (7, 'rgb(255,168,255)'),
                     'Tsy': (8, 'rgb(255,232,196)'), 'Tby': (9, 'rgb(255,186,227)'),
                     'Tb': (10, 'rgb(255,158,227)'), 'Tsv': (11, 'rgb(251,196,123)'),
                     'Tsm': (12, 'rgb(226,184,84)'), 'Tv': (13, 'rgb(253,193,88)'),
                     'Tg': (14, 'rgb(251,186,123)'), 'Ti': (15, 'rgb(253,158,105)'),
                     'TXgn': (16, 'rgb(251,154,170)'), 'Tso': (17, 'rgb(238,215,146)'),
                     'TKgm': (18, 'rgb(255,135,145)'), 'TKg': (19, 'rgb(253,119,101)'),
                     'Kv': (20, 'rgb(209,255,135)'), 'KJo': (21, 'rgb(186,255,158)'),
                     'Kmv': (22, 'rgb(232,255,168)'), 'Ks': (23, 'rgb(179,251,133)'),
                     'Jm': (24, 'rgb(120,240,119)'), 'Yg': (25, 'rgb(202,139,79)'),
                     'Jsv': (26, 'rgb(125,255,178)'), 'Jg': (27, 'rgb(228,95,176)'),
                     'J^': (28, 'rgb(178,255,240)'), '}|': (29, 'rgb(129,253,128)'),
                     'Js': (30, 'rgb(121,226,157)'), 'Jgc': (31, 'rgb(109,241,168)'),
                     '^c': (32, 'rgb(125,255,209)'), '^m': (33, 'rgb(144,222,212)'),
                     '|': (34, 'rgb(123,236,251)'), 'P': (35, 'rgb(168,240,255)'),
                     'P*': (36, 'rgb(158,217,255)'), 'M_': (37, 'rgb(196,217,240)'),
                     'Ys': (38, 'rgb(186,178,255)'), 'Yd': (39, 'rgb(213,151,208)'),
                     'YXg': (40, 'rgb(209,161,119)'), 'Xg': (41, 'rgb(186,130,123)'),
                     'Xms': (42, 'rgb(185,196,185)'), 'Xq': (43, 'rgb(146,179,95)'),
                     'Xmv': (44, 'rgb(144,179,136)'), 'Xm': (45, 'rgb(178,178,95)'),
                     'NODATA': (46, 'rgb(0,0,0)')}

        task = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                            .with_chip_size(366) \
                            .with_classes(classes) \
                            .with_chip_options(
                                chips_per_scene=chips_per_scene,
                                debug_chip_probability=0.2,
                                negative_survival_probability=1.0) \
                            .build()

        backend = rv.BackendConfig.builder(rv.TF_DEEPLAB) \
                                  .with_task(task) \
                                  .with_model_defaults(model_type) \
                                  .with_train_options(sync_interval=600) \
                                  .with_num_steps(num_steps) \
                                  .with_batch_size(batch_size) \
                                  .with_debug(debug) \
                                  .build()

        train_scenes = [build_scene(task, data_uri, id, channel_order)
                      for id in train_ids]
        val_scenes = [build_scene(task, data_uri, id, channel_order)
                      for id in val_ids]

        augmentor = rv.AugmentorConfig(rv.NODATA_AUGMENTOR) \
                                  .with_probability(0.3) \ 
                                  .build()

        dataset = rv.DatasetConfig.builder() \
                                  .with_augmentor(augmentor) \
                                  .with_train_scenes(train_scenes) \
                                  .with_validation_scenes(val_scenes) \
                                  .build()


        experiment = rv.ExperimentConfig.builder() \
                                        .with_id('geo-seg-xcept') \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_dataset(dataset) \
                                        .with_root_uri(root_uri) \
                                        .with_stats_analyzer() \
                                        .build()

        return experiment


if __name__ == '__main__':
    rv.main()
