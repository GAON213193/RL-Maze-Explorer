"""Save after PPO updates; log per-episode outcomes without oracle inputs."""
import csv
import json
from pathlib import Path
import shutil
from stable_baselines3.common.callbacks import BaseCallback
from .cloud import make_bundle, publish


def atomic_copy(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + '.tmp')
    shutil.copy2(source, temp)
    temp.replace(destination)


class Checkpoints(BaseCallback):
    def __init__(self, output, interval, model_path=None, cloud_tag=None):
        super().__init__()
        self.output, self.interval = output, interval
        self.model_path, self.cloud_tag = model_path, cloud_tag
        self.last_save = None
        self.last_uploaded = None
        self.episodes = self.successes = self.depletions = self.turns = 0
        self.recent = []
        self.stalls = self.unique_cells = self.forward_moves = 0
        self.action_counts = [0, 0, 0, 0]

    def _on_training_start(self):
        self.last_save = self.model.num_timesteps
        with (self.output / 'episodes.csv').open('w', newline='') as f:
            csv.writer(f).writerow(['total_steps','reason','turns','success','battery_depleted','unique_cells','forward_moves'])

    def _on_step(self):
        for action in self.locals['actions']:
            self.action_counts[int(action)] += 1
        for done, info in zip(self.locals['dones'], self.locals['infos']):
            if done:
                success = bool(info.get('is_success'))
                depleted = info.get('reason') in ('battery_depleted', 'insufficient_energy')
                turns = info.get('turns', 0)
                self.stalls += info.get('reason') == 'stalled'
                self.unique_cells += info.get('unique_cells', 0)
                self.forward_moves += info.get('forward_moves', 0)
                self.episodes += 1
                self.successes += success
                self.depletions += depleted
                self.turns += turns
                self.recent.append(int(success)); self.recent = self.recent[-100:]
                with (self.output / 'episodes.csv').open('a', newline='') as f:
                    csv.writer(f).writerow([self.model.num_timesteps, info.get('reason'), turns,
                                           int(success), int(depleted), info.get('unique_cells'), info.get('forward_moves')])
        return True

    def _on_rollout_start(self):
        # This is after the previous update, not midway through rollout collection.
        if self.model.num_timesteps - self.last_save >= self.interval:
            self.save()

    def save(self):
        temp = self.output / 'saving.zip'
        self.model.save(temp)
        temp.replace(self.output / 'model.zip')
        metrics = dict(total_steps=self.model.num_timesteps, episodes=self.episodes,
                       stalls=self.stalls, action_counts=self.action_counts,
                       mean_unique_cells=self.unique_cells/self.episodes if self.episodes else None,
                       mean_forward_moves=self.forward_moves/self.episodes if self.episodes else None,
                       successes=self.successes, battery_depletions=self.depletions,
                       mean_turns=self.turns/self.episodes if self.episodes else None,
                       success_rate=self.successes/self.episodes if self.episodes else None,
                       recent_100_success_rate=sum(self.recent)/len(self.recent) if self.recent else None)
        (self.output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        if self.model_path:
            atomic_copy(self.output / 'model.zip', self.model_path)
            atomic_copy(self.output / 'config.json', Path(self.model_path).with_suffix('.json'))
        self.last_save = self.model.num_timesteps
        print('Local backup saved:', self.last_save, metrics, flush=True)
        if self.cloud_tag:
            bundle = make_bundle(self.output, self.last_save)
            try:
                publish(bundle, self.cloud_tag)
                self.last_uploaded = self.last_save
                print('GitHub saved:', self.cloud_tag, bundle.name, flush=True)
            except (RuntimeError, OSError) as exc:
                print('GITHUB UPLOAD FAILED. Local backup retained:', exc, flush=True)
