argus_inference
===============

The decoder of the `Argus Cybernetics`_ stack: a ROS 2 node that turns the
fabric codec's spike features, one frame per 50 ms bin, into a velocity
command on ``/cmd_vel``.

The whole stack, the one command that runs it on the board, and the results
are in the `argus_bringup README`_. This site documents the package itself.

.. _Argus Cybernetics: https://github.com/Max-Gabriel-Susman/argus_bringup
.. _argus_bringup README: https://github.com/Max-Gabriel-Susman/argus_bringup/blob/main/README.md

.. toctree::
   :maxdepth: 2

   inference_node
