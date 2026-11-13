"""Synthetic CPU verification of CLI export and shared ensemble formulas."""
import ast
from contextlib import redirect_stdout,redirect_stderr
from copy import deepcopy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy.special import softmax
import torch

import predict
from common import HERE,CLASSES,LABELS,read,save,sha
from evaluate import apply_residual
from inference.engine_n import square_context


class PredictionChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='predict_cpu_',dir=HERE/'data_review')
        self.folder=Path(self.temp.name)
        self.source=self.folder/'source.png'
        self.image=Image.new('RGB',(200,180),(80,140,190))
        self.image.save(self.source)
        self.spec={'id':'synthetic_closed','architecture':'dinov2_tail','form':'closed',
                   'temperature':1.7,'models':[{'seed':n} for n in (131,271,719)]}
        self.boxes=[{'xyxy':[5.,10.,35.,50.],'confidence':.4,
                     'species_info':{'accepted':False,'birdness_score':.3,'feature_index':0}},
                    {'xyxy':[110.,100.,140.,140.],'confidence':.8,
                     'species_info':{'accepted':True,'birdness_score':.96,'feature_index':1}}]
        self.crops=[square_context(self.image,box['xyxy']) for box in self.boxes]

    def tearDown(self):
        self.temp.cleanup()

    def test_cli_exports_all_original_boxes_primary_and_gate(self):
        probabilities=np.zeros((2,len(CLASSES)),dtype=np.float64)
        probabilities[0,0]=1.;probabilities[1,1]=1.
        destination=self.folder/'output'
        original=sha
        def bound_hash(path):
            return 'frozen' if Path(path)==HERE/'selection.json' else original(path)
        fake=(deepcopy(self.boxes),1,self.crops,np.zeros((2,768),np.float32),None,{'numeric_mode':'cpu_test'})
        with patch.object(predict,'select_candidate',return_value=({'deployment_candidate_id':'synthetic_closed'},self.spec,'frozen')), \
             patch.object(predict,'normal_detection',return_value=fake) as detector, \
             patch.object(predict,'species_probabilities',return_value=(probabilities,CLASSES,{})), \
             patch.object(predict,'sha',side_effect=bound_hash), \
             patch.object(sys,'argv',['predict','--image',str(self.source),'--output',str(destination),
                                      '--candidate','synthetic_closed','--device','cpu']),redirect_stdout(io.StringIO()):
            predict.main()
        detector.assert_called_once()
        result=read(destination/'result.json')
        self.assertEqual(result['candidate_count'],2)
        self.assertFalse(result['count_is_unique_individual_count'])
        self.assertEqual(result['primary_index'],1)
        self.assertEqual([r['xyxy'] for r in result['candidates']],[r['xyxy'] for r in self.boxes])
        self.assertEqual([r['accepted_original_E'] for r in result['candidates']],[False,True])
        self.assertEqual([r['primary'] for r in result['candidates']],[False,True])
        self.assertEqual(result['candidates'][0]['display_name'],'鸟（种类不确定）')
        for item,expected in zip(result['candidates'],self.crops):
            with Image.open(destination/item['crop']) as actual:
                self.assertEqual(actual.size,expected.size)
                self.assertEqual(actual.tobytes(),expected.tobytes())

    def test_empty_detection_exports_without_new_encoder_calls(self):
        folder=self.folder/'empty';folder.mkdir()
        summary=predict.export(self.image,self.source,folder,{},self.spec,'frozen',[],None,[],
                               np.empty((0,len(CLASSES))),CLASSES,{}, {})
        self.assertEqual(summary['candidate_count'],0)
        self.assertIsNone(summary['primary_index'])
        self.assertEqual(summary['candidates'],[])
        self.assertEqual(list(folder.glob('bird_*.png')),[])
        self.assertTrue((folder/'annotated.jpg').exists())

    def test_closed_uses_mean_logits_then_temperature(self):
        rng=np.random.default_rng(29)
        logits=[rng.normal(size=(2,len(CLASSES))) for _ in range(3)]
        with patch('scoring.family_logits',return_value=({'photo':logits},[])), \
             patch.object(predict,'encode_dino',return_value=(np.zeros((2,257,384),np.float32),None)), \
             patch.object(predict,'encode_bio25') as unused:
            values,classes,_=predict.species_probabilities(self.spec,self.crops,np.zeros((2,768),np.float32),None,'cpu')
        unused.assert_not_called()
        np.testing.assert_array_equal(classes,CLASSES)
        np.testing.assert_allclose(values,softmax(np.mean(logits,axis=0)/1.7,axis=1),rtol=0,atol=0)

    def test_full_residual_uses_shared_raw_base_and_preserves_outside_winner(self):
        rng=np.random.default_rng(73)
        logits=[rng.normal(size=(2,len(CLASSES))) for _ in range(3)]
        base=np.full((2,len(LABELS)),1e-10)
        outside=next(i for i in range(len(LABELS)) if i not in CLASSES)
        base[0,CLASSES[0]]=.7;base[0,CLASSES[1]]=.3
        base[1,outside]=.99;base[1,CLASSES[0]]=.01
        base/=base.sum(1,keepdims=True)
        spec={**self.spec,'architecture':'bioclip_tail','form':'full_residual','alpha':.5}
        with patch('scoring.family_logits',return_value=({'photo':logits},[])), \
             patch('evaluate.raw_base',return_value={'deployed_BioCLIP2_2_5':base}) as raw, \
             patch.object(predict,'encode_bio25',return_value=np.zeros((2,1024))), \
             patch.object(predict,'encode_dino',return_value=(None,np.zeros((2,384)))):
            values,classes,_=predict.species_probabilities(spec,self.crops,np.zeros((2,768)),np.zeros((2,257,1024)),'cpu')
        raw.assert_called_once()
        np.testing.assert_allclose(values,apply_residual(base,np.mean(logits,axis=0),.5,1.7),rtol=0,atol=0)
        np.testing.assert_array_equal(values[1],base[1])
        np.testing.assert_array_equal(classes,np.arange(len(LABELS)))

    def test_nonempty_folder_is_rejected_before_model_selection(self):
        destination=self.folder/'nonempty';destination.mkdir()
        (destination/'keep').write_text('unchanged')
        with patch.object(predict,'select_candidate') as selector, \
             patch.object(sys,'argv',['predict','--image',str(self.source),'--output',str(destination),'--candidate','x']), \
             redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as stopped:
            predict.main()
        self.assertEqual(stopped.exception.code,2)
        selector.assert_not_called()
        self.assertEqual((destination/'keep').read_text(),'unchanged')

    def test_source_ast_parses_and_has_main(self):
        tree=ast.parse((HERE/'predict.py').read_text(encoding='utf-8'))
        names={node.name for node in tree.body if isinstance(node,ast.FunctionDef)}
        self.assertTrue({'main','normal_detection','bio_prefix','encode_dino','species_probabilities','export'}<=names)


if __name__=='__main__':
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PredictionChecks))
    save(HERE/'data_review/predict_cpu_review.json',{'status':'passed' if result.wasSuccessful() else 'failed',
         'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
         'CUDA_initialized':torch.cuda.is_initialized(),'model_inference_performed':False,
         'scope':'Synthetic CPU exports/formulas and AST; real YOLO/backbones are mocked, not claimed as tested',
         'predict_sha256':sha(HERE/'predict.py'),'test_source_sha256':sha(__file__),'output':stream.getvalue()})
    print(stream.getvalue(),flush=True)
    if not result.wasSuccessful():
        raise SystemExit(1)
