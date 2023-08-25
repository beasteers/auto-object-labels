import os
import shutil




def flatten_paths(root_dir, out_dir, *files):
    os.makedirs(out_dir, exist_ok=True)
    for f in files:
        f2 = os.path.join(out_dir, os.path.relpath(f, root_dir).replace(os.sep, '_'))
        print(f, '->', f2)
        shutil.copyfile(f, f2)


if __name__ == '__main__':
    import fire
    fire.Fire()

# python util.py flatten_paths ~/Downloads/Milly/CookBookPhase2_v5/ ~/Downloads/Milly-flat ~/Downloads/Milly/CookBookPhase2_v5/*/*/*/*/*.mp4