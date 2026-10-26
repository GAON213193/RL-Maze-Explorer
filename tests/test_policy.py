import tempfile
import unittest
import numpy as np
import torch
from stable_baselines3 import PPO
from training.policy import ChargeMaskedPolicy
from environment.gym_v3 import RangeChargeEnv
from environment.robot import Config

class PolicyTests(unittest.TestCase):
    def test_mask_sampling_update_and_reload(self):
        torch.set_num_threads(1)
        env=RangeChargeEnv(Config(size=7,charge_turns=2,charge_threshold=.5),0)
        model=PPO(ChargeMaskedPolicy,env,n_steps=8,batch_size=8,n_epochs=1,device='cpu')
        obs,_=env.reset(seed=42)
        # Force a huge preference for charging: masking must still win.
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias[:]=torch.tensor([0.,0.,0.,100.])
        for battery,at_start,allowed in [(.51,True,False),(.5,True,True),(.49,True,True),(.1,False,False)]:
            obs['state'][8]=battery
            obs['state'][0:2]=obs['state'][2:4]
            if not at_start:obs['state'][0]+=0.1
            tensor,_=model.policy.obs_to_tensor(obs)
            dist=model.policy.get_distribution(tensor)
            self.assertEqual(float(dist.distribution.probs[0,3])>0,allowed)
            for deterministic in (False,True):
                action,_=model.predict(obs,deterministic=deterministic)
                self.assertEqual(int(action)==3,allowed)
            action,value,logp=model.policy(tensor)
            _,lp,entropy=model.policy.evaluate_actions(tensor,action)
            torch.testing.assert_close(logp,lp)
            self.assertTrue(torch.isfinite(entropy).all())
        model.learn(16)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.policy.parameters()))
        with tempfile.TemporaryDirectory() as d:
            model.save(d+'/model')
            restored=PPO.load(d+'/model',env=env)
            self.assertIsInstance(restored.policy,ChargeMaskedPolicy)
            obs,_=env.reset(seed=1)
            self.assertNotEqual(int(restored.predict(obs)[0]),3)
            old=PPO('MultiInputPolicy',env,n_steps=8,batch_size=8,n_epochs=1)
            old.save(d+'/old')
            migrated=PPO.load(d+'/old',env=env,custom_objects={'policy_class':ChargeMaskedPolicy})
            for key,value in old.policy.state_dict().items():
                torch.testing.assert_close(value,migrated.policy.state_dict()[key])
            migrated.learn(8)
            self.assertNotEqual(int(migrated.predict(obs)[0]),3)
