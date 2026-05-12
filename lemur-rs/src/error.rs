//! Error types for the Lemur implementation.

#[derive(Debug, thiserror::Error)]
pub enum LemurError {
    #[error("verification failed")]
    VerifyFailed,
    #[error("aggregation failed after {0} attempts")]
    AggregationFailed(usize),
    #[error("invalid encoding: {0}")]
    InvalidEncoding(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
