Model-owned Doom decisions: Laya and Jev choose the action, target and weapon;
the shared motor controller executes that choice.

The release includes a Doom-adapted Laya v3 checkpoint and side-by-side videos.
On FreeDoom MAP01, skill 3, seed 48: original Laya timed out after 180 seconds;
adapted Laya reached MAP02 in 61.429 seconds; Jev reached MAP02 in 68.971 seconds.
Both finishers survived. Laya was fine-tuned for this task; Jev was not.
One map and one final seed do not establish general model superiority.

HTTP median including RTT: adapted Laya 108 ms, Jev 357 ms.
Jev API cost for the final run: $0.004580184. Local compute was not priced.
Training instructions and frozen datasets are included in the source repository.

The model archive contains Apache-2.0, attribution, a model card and SHA256SUMS.
Verify laya-doom-v3.tar.sha256 before extracting the archive into checkpoints/.

Model: https://huggingface.co/azalio/laya-doom-v3
Training guide: https://github.com/azalio/doomLaya/blob/main/TRAINING.md
