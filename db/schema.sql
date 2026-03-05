--------------------------------------------------------------
-- Config
ALTER DATABASE zoo_vision SET timezone TO 'Europe/Zurich';

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
    name            text NOT NULL,
	color           text NOT NULL
);

INSERT INTO identities(id, name, color)
	VALUES 
	(0, 'Invalid', '#777777'),
	(1, 'Chandra', '#73BF69'),
	(2, 'Indi', '#F2CC0C'),
	(3, 'Farha', '#5794F2'),
	(4, 'Panang', '#FF9830'),
	(5, 'Thai', '#F2495C');

CREATE TABLE behaviours (
    id              int PRIMARY KEY,
    name            text NOT NULL,
    column_name     text NOT NULL
);

INSERT INTO behaviours(id, name)
	VALUES 
	(0, 'Invalid', 'invalid'),
	(1, 'Standing', 'standing'),
	(2, 'Sleep-Left side', 'sleep_left_side'),
	(3, 'Sleep-Right side', 'sleep_right_side'),
	(4, 'Walking', 'walking'),
	(5, 'Stereotypy', 'stereotypy'),
	(6, 'No observation', 'no_observation');

--------------------------------------------------------------
-- Real-time inputs

CREATE TABLE tracks (
    id              int PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    camera_id		int NOT NULL REFERENCES cameras,
    start_time      timestamptz NOT NULL,
    end_time        timestamptz NULL,  	-- Only valid after track is closed
    frame_count     int NULL,			-- Only valid after track is closed
    identity_id     int NULL REFERENCES identities, -- ...  track is closed
	track_filename  text NULL
);

CREATE TABLE identity_probs (
    track_id        int NOT NULL REFERENCES tracks,
    identity_id     int NOT NULL REFERENCES identities,
    prob            real NOT NULL,
	PRIMARY KEY (track_id, identity_id)
);

CREATE TABLE observations (
    track_id        int NOT NULL REFERENCES tracks,
    time            timestamptz NOT NULL,
    location        point NOT NULL,
    behaviour_id	int NOT NULL REFERENCES behaviours,
    PRIMARY KEY (track_id, time)
);
CREATE INDEX ON observations (time);


--------------------------------------------------------------
-- Processed
CREATE TABLE ethogram (
    id              int PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    identity_id     int NOT NULL REFERENCES identities,
    behaviour_id    int NOT NULL REFERENCES behaviours,
    start_dt        timestamptz NOT NULL,
    end_dt          timestamptz NOT NULL
);
CREATE INDEX ON ethogram (identity_id, start_dt);
CREATE INDEX ON ethogram (start_dt);

CREATE TABLE ethogram_summary (
     identity_id      INT NOT NULL,
     time_bined       TIMESTAMPTZ NOT NULL,
     invalid          INT NOT NULL,
     standing         INT NOT NULL,
     sleep_left_side  INT NOT NULL,
     sleep_right_side INT NOT NULL,
     walking          INT NOT NULL,
     stereotypy       INT NOT NULL,
     no_observation   INT NOT NULL,
	 PRIMARY KEY (identity_id, time_bined)
) 

CREATE TABLE summary_per_behaviour (
	identity_id      int NOT NULL REFERENCES identities,
	time_bined       timestamptz NOT NULL, -- Aggregated to 1s
	invalid          int,
	standing         int,
	sleep_left_side  int,
	sleep_right_side int,
	walking          int,
	stereotypy       int,
	no_observation   int,
	PRIMARY KEY (identity_id, time_bined)
);
CREATE INDEX ON summary_per_behaviour (identity_id, time);
CREATE INDEX ON summary_per_behaviour (time);

CREATE TABLE summary_per_visibility (
	identity_id     int NOT NULL REFERENCES identities,
	time_bined      timestamptz NOT NULL, -- Aggregated to seconds
	location        point NULL,
	PRIMARY KEY (identity_id, time_bined)
);

--------------------------------------------------------------
-- Extensions
CREATE EXTENSION IF NOT EXISTS tablefunc; -- enables crosstab for pivoting

--------------------------------------------------------------
-- Permissions
CREATE USER zoo_admin SUPERUSER PASSWORD 'asdf';
GRANT CONNECT ON DATABASE zoo_vision TO zoo_admin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO zoo_admin;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public to zoo_vision;

CREATE USER zoo_vision PASSWORD 'asdf';
GRANT CONNECT ON DATABASE zoo_vision TO zoo_vision;
GRANT INSERT ON ALL TABLES IN SCHEMA public TO zoo_vision;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO zoo_vision;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public to zoo_vision;

CREATE USER grafanareader WITH PASSWORD 'asdf';
GRANT CONNECT ON DATABASE zoo_vision TO grafanareader;
GRANT USAGE ON SCHEMA public TO grafanareader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO grafanareader;
