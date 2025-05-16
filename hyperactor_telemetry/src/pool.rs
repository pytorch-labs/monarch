use std::sync::mpmc::Receiver;
use std::sync::mpmc::Sender;
use std::sync::mpmc::sync_channel;
use std::sync::mpsc::TryRecvError;

/// A basic thread-safe pool of objects with a fixed capacity.
/// The implementation uses a simple mpmc channel to store the
/// objects.
#[derive(Debug)]
pub(crate) struct Pool<T: Default> {
    sender: Sender<T>,
    receiver: Receiver<T>,
}

impl<T: Default> Pool<T> {
    pub(crate) fn new(cap: usize) -> Self {
        let (sender, receiver) = sync_channel(cap);
        Self { sender, receiver }
    }

    pub(crate) fn get(&self) -> T {
        match self.receiver.try_recv() {
            Ok(val) => val,
            Err(TryRecvError::Empty) => Default::default(),
            Err(TryRecvError::Disconnected) => panic!("channel disconnected"),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn put(&self, value: T) {
        let _ = self.sender.try_send(value);
    }
}

// Manual implementation to avoid demanding T: Clone
impl<T: Default> Clone for Pool<T> {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
            receiver: self.receiver.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let pool: Pool<u32> = Pool::new(2);

        assert_eq!(pool.get(), 0);
        pool.put(1);
        assert_eq!(pool.get(), 1);
        pool.put(2);
        assert_eq!(pool.get(), 2);
        assert_eq!(pool.get(), 0);
        pool.put(3);
        pool.put(4);
        pool.put(5);
        assert_eq!(pool.get(), 3);
        assert_eq!(pool.get(), 4);
        assert_eq!(pool.get(), 0);

        pool.put(3);
        assert_eq!(pool.clone().get(), 3);
        pool.clone().put(123);
        assert_eq!(pool.get(), 123);
    }
}
