docker run -it \
	--name=conda_ros \
	--volume="/home/sanusha/.ssh/id_ed25519:/home/sanusha/.ssh/id_ed25519:ro" \
	--volume="/home/sanusha/data_driven_lca_codes:/home/sanusha/codes" \
	conda_lca /bin/bash
