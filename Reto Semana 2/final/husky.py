from robot import HuskyA200

husky = HuskyA200()
v, omega = husky.forward_kinematics(4,4,2,2)

print(v, omega)