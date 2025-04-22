--------------------------------------------------------------
-- Config
CREATE TABLE cameras (
    id              int PRIMARY KEY,
    name            text NOT NULL
);

INSERT INTO cameras(id, name)
	VALUES 
	(0, 'zag_elp_cam_016'),
	(1, 'zag_elp_cam_017'),
	(2, 'zag_elp_cam_018'),
	(3, 'zag_elp_cam_019');

CREATE TABLE identities (
    id              int PRIMARY KEY,
    name            text NOT NULL
);

INSERT INTO identities(id, name)
	VALUES 
	(0, 'Invalid'),
	(1, 'Chandra'),
	(2, 'Indi'),
	(3, 'Fahra'),
	(4, 'Panang'),
	(5, 'Thai');

CREATE TABLE behaviours (
    id              int PRIMARY KEY,
    name            text NOT NULL
);

INSERT INTO behaviours(id, name)
	VALUES 
	(0, 'Invalid'),
	(1, 'Standing'),
	(2, 'Sleep-Left side'),
	(3, 'Sleep-Right side');

--------------------------------------------------------------
-- Real-time inputs

CREATE TABLE tracks (
    id              int PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    camera_id		int NOT NULL REFERENCES cameras,
    start_time      timestamp NOT NULL,
    end_time        timestamp NULL,  	-- Only valid after track is closed
    frame_count     int NULL,			-- Only valid after track is closed
    identity_id     int NULL REFERENCES identities  -- ...  track is closed
);

CREATE TABLE identity_probs (
    track_id        int NOT NULL REFERENCES tracks,
    identity_id     int NOT NULL REFERENCES identities,
    prob            real NOT NULL,
	PRIMARY KEY (track_id, identity_id)
);

CREATE TABLE observations (
    track_id        int NOT NULL REFERENCES tracks,
    time            timestamp NOT NULL,
    location        point NOT NULL,
    behaviour_id	int NOT NULL REFERENCES behaviours,
    PRIMARY KEY (track_id, time)
);

--------------------------------------------------------------
-- Processed
CREATE TABLE summary_per_behaviour (
	identity_id     int NOT NULL REFERENCES identities,
	time            timestamp NOT NULL, -- Aggregated to minutes
	is_standing     bool NOT NULL,    
	is_on_left_side bool NULL,
	PRIMARY KEY (identity_id, time)
);

CREATE TABLE summary_per_visibility (
	identity_id     int NOT NULL REFERENCES identities,
	time            timestamp NOT NULL, -- Aggregated to seconds
	location        point NULL,
	PRIMARY KEY (identity_id, time)
);

--------------------------------------------------------------
-- Permissions
CREATE USER zoo_vision PASSWORD 'asdf';
GRANT CONNECT ON DATABASE zoo_vision TO zoo_vision;
GRANT INSERT ON tracks,observations,summary_per_behaviour,summary_per_visibility TO zoo_vision;

CREATE USER grafanareader WITH PASSWORD 'asdf';
GRANT CONNECT ON DATABASE zoo_vision TO grafanareader;
GRANT USAGE ON SCHEMA public TO grafanareader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO grafanareader;
