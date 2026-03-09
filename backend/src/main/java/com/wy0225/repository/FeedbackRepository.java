package com.wy0225.repository;

import com.wy0225.entity.Feedback;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface FeedbackRepository extends JpaRepository<Feedback, Long> {
    List<Feedback> findByUserIdOrderByCreatedAtDesc(Long userId);

    @org.springframework.data.jpa.repository.Query("SELECT f FROM Feedback f WHERE " +
            "(:userId IS NULL OR f.userId = :userId) AND " +
            "(:modelType IS NULL OR f.modelType = :modelType) " +
            "ORDER BY f.createdAt DESC")
    List<Feedback> findAllWithFilters(
            @org.springframework.data.repository.query.Param("userId") Long userId,
            @org.springframework.data.repository.query.Param("modelType") String modelType);
}
