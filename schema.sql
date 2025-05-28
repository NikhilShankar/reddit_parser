-- Database schema for reddit_mindfulness
-- Run this in your MySQL database

USE reddit_mindfulness;

-- Create posts table
CREATE TABLE IF NOT EXISTS posts (
    id VARCHAR(20) PRIMARY KEY,
    title TEXT NOT NULL,
    author VARCHAR(50),
    content TEXT,
    url TEXT,
    score INT DEFAULT 0,
    upvote_ratio DECIMAL(3,2),
    num_comments INT DEFAULT 0,
    created_utc TIMESTAMP,
    subreddit VARCHAR(50),
    is_self BOOLEAN DEFAULT FALSE,
    selftext TEXT,
    permalink TEXT,
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create comments table
CREATE TABLE IF NOT EXISTS comments (
    id VARCHAR(20) PRIMARY KEY,
    post_id VARCHAR(20),
    author VARCHAR(50),
    body TEXT,
    score INT DEFAULT 0,
    created_utc TIMESTAMP,
    parent_type ENUM('post', 'comment') DEFAULT 'post',
    parent_id VARCHAR(20),
    permalink TEXT,
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE,
    INDEX idx_post_id (post_id),
    INDEX idx_author (author),
    INDEX idx_created_utc (created_utc)
);