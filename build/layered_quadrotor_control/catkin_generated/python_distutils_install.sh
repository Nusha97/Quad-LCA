#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/anusha/cleanup_ws/src/layered_quadrotor_control"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/anusha/cleanup_ws/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/anusha/cleanup_ws/install/lib/python3/dist-packages:/home/anusha/cleanup_ws/build/layered_quadrotor_control/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/anusha/cleanup_ws/build/layered_quadrotor_control" \
    "/usr/bin/python3" \
    "/home/anusha/cleanup_ws/src/layered_quadrotor_control/setup.py" \
     \
    build --build-base "/home/anusha/cleanup_ws/build/layered_quadrotor_control" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/anusha/cleanup_ws/install" --install-scripts="/home/anusha/cleanup_ws/install/bin"
