# Example: training CTRGCN on NTU RGB+D 120 cross subject with GPU 0
#nohup python main.py --config config/nturgbd-cross-subject/joint.yaml --device 0 1 2 3 &
#nohup python main.py --config config/nturgbd-cross-subject/motion.yaml --device 0 1 2 3 &
#nohup python main.py --config config/nturgbd-cross-subject/bone.yaml --device 4 5 6 7 &
#nohup python main.py --config config/nturgbd-cross-subject/bone_motion.yaml --device 4 5 6 7 &


nohup python main.py --config work_dir/ntu60/xsub/ctrgcn_joint/config.yaml --work-dir work_dir/ntu60/xsub/ctrgcn_joint --phase test --save-score True --weights work_dir/ntu60/xsub/ctrgcn_joint/runs-65-40690.pt --device 0 1 2 3 &

nohup python main.py --config work_dir/ntu60/xsub/ctrgcn_bone/config.yaml --work-dir work_dir/ntu60/xsub/ctrgcn_bone --phase test --save-score True --weights work_dir/ntu60/xsub/ctrgcn_bone/runs-65-40690.pt --device 4 5 6 7 &

nohup python main.py --config work_dir/ntu60/xsub/ctrgcn_motion/config.yaml --work-dir work_dir/ntu60/xsub/ctrgcn_motion --phase test --save-score True --weights work_dir/ntu60/xsub/ctrgcn_motion/runs-65-40690.pt --device 0 1 2 3 &

nohup python main.py --config work_dir/ntu60/xsub/ctrgcn_bone_motion/config.yaml --work-dir work_dir/ntu60/xsub/ctrgcn_bone_motion --phase test --save-score True --weights work_dir/ntu60/xsub/ctrgcn_bone_motion/runs-65-40690.pt --device 4 5 6 7 &


#python ensemble.py --dataset ntu60/xsub --joint-dir work_dir/ntu60/xsub/ctrgcn_joint --bone-dir work_dir/ntu60/xsub/ctrgcn_bone --joint-motion-dir work_dir/ntu60/xsub/ctrgcn_motion --bone-motion-dir work_dir/ntu60/xsub/ctrgcn_bone_motion




#nohup python main.py --config config/ucla/joint.yaml --work-dir work_dir/ucla/ctrgcn_joint --device 4 &
#nohup python main.py --config config/ucla/motion.yaml --work-dir work_dir/ucla/ctrgcn_motion --device 5 &
#nohup python main.py --config config/ucla/bone.yaml --work-dir work_dir/ucla/ctrgcn_bone --device 6 &
#nohup python main.py --config config/ucla/bone_motion.yaml --work-dir work_dir/ucla/ctrgcn_bone_motion --device 7 &

nohup python main.py --config work_dir/ucla/ctrgcn_joint/config.yaml --work-dir work_dir/ucla/ctrgcn_joint --phase test --save-score True --weights work_dir/ucla/ctrgcn_joint/runs-65-20670.pt --device 4 &

nohup python main.py --config work_dir/ucla/ctrgcn_bone/config.yaml --work-dir work_dir/ucla/ctrgcn_bone --phase test --save-score True --weights work_dir/ucla/ctrgcn_bone/runs-65-20670.pt --device 5 &

nohup python main.py --config work_dir/ucla/ctrgcn_motion/config.yaml --work-dir work_dir/ucla/ctrgcn_motion --phase test --save-score True --weights work_dir/ucla/ctrgcn_motion/runs-65-20670.pt --device 6 &

nohup python main.py --config work_dir/ucla/ctrgcn_bone_motion/config.yaml --work-dir work_dir/ucla/ctrgcn_bone_motion --phase test --save-score True --weights work_dir/ucla/ctrgcn_bone_motion/runs-65-20670.pt --device 7 &

#python ensemble.py --dataset NW-UCLA --joint-dir work_dir/ucla/ctrgcn_joint --bone-dir work_dir/ucla/ctrgcn_bone --joint-motion-dir work_dir/ucla/ctrgcn_motion --bone-motion-dir work_dir/ucla/ctrgcn_bone_motion

